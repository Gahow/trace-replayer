use super::{LLMApi, RequestError, MODEL_NAME};
use futures_util::TryStreamExt;
use reqwest::Response;
use serde_json::json;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    time::{timeout as tokio_timeout, Instant as TokioInstant},
};
use tokio_util::io::StreamReader;

#[derive(Copy, Clone)]
pub struct DashScopeApi;

fn _extract_chunk_text(payload: &Value) -> Option<String> {
    if let Some(text) = payload
        .get("output")
        .and_then(|output| output.get("text"))
        .and_then(Value::as_str)
    {
        return Some(text.to_string());
    }
    if let Some(choice) = payload
        .get("output")
        .and_then(|output| output.get("choices"))
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    {
        if let Some(text) = choice.get("text").and_then(Value::as_str) {
            return Some(text.to_string());
        }
        if let Some(content) = choice
            .get("message")
            .and_then(|message| message.get("content"))
        {
            if let Some(text) = content.as_str() {
                return Some(text.to_string());
            }
            if let Some(parts) = content.as_array() {
                let merged = parts
                    .iter()
                    .filter_map(|item| item.get("text").and_then(Value::as_str))
                    .collect::<Vec<&str>>()
                    .join("");
                if !merged.is_empty() {
                    return Some(merged);
                }
            }
        }
    }
    None
}

#[async_trait::async_trait]
impl LLMApi for DashScopeApi {
    const AIBRIX_PRIVATE_HEADER: bool = false;
    const DASHSCOPE_SSE_HEADER: bool = true;

    fn request_json_body(prompt: String, output_length: u64, stream: bool) -> String {
        let mut parameters = json!({
            "max_length": output_length
        });
        if stream {
            parameters["incremental_output"] = Value::Bool(true);
        }
        let json_body = json!({
            "model": MODEL_NAME.get().unwrap().as_str(),
            "input": {
                "prompt": prompt
            },
            "parameters": parameters
        });
        json_body.to_string()
    }

    async fn parse_response(
        response: Response,
        _stream: bool,
        timeout_duration: Duration,
    ) -> Result<BTreeMap<String, String>, RequestError> {
        let mut result = BTreeMap::new();
        result.insert("status".to_string(), response.status().as_str().to_string());
        if !_stream {
            return Ok(result);
        }
        if !response.status().is_success() {
            return Ok(result);
        }

        let stream = response.bytes_stream();
        let stream_reader = StreamReader::new(
            stream.map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error)),
        );
        let mut reader = BufReader::new(stream_reader);
        let mut line = String::new();
        let mut first_token_time: Option<TokioInstant> = None;
        let mut last_token_time: Option<TokioInstant> = None;
        let mut chunk_intervals: Vec<f64> = Vec::new();
        let start_time = TokioInstant::now();

        loop {
            if start_time.elapsed() > timeout_duration {
                return Err(RequestError::Timeout);
            }
            let remaining_duration = timeout_duration - start_time.elapsed();
            let read_future = reader.read_line(&mut line);
            match tokio_timeout(remaining_duration, read_future).await {
                Ok(Ok(0)) => break,
                Ok(Ok(_)) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        line.clear();
                        continue;
                    }
                    if !trimmed.starts_with("data: ") {
                        line.clear();
                        continue;
                    }
                    let data_str = &trimmed[6..];
                    if data_str == "[DONE]" {
                        break;
                    }
                    if let Ok(payload) = serde_json::from_str::<Value>(data_str) {
                        let chunk_text = _extract_chunk_text(&payload).unwrap_or_default();
                        if !chunk_text.is_empty() {
                            let now = TokioInstant::now();
                            if first_token_time.is_none() {
                                first_token_time = Some(now);
                                let first_token_duration =
                                    now.duration_since(start_time).as_secs_f64() * 1000.0;
                                result.insert(
                                    "first_token_time".to_string(),
                                    format!("{first_token_duration:.3}"),
                                );
                            } else if let Some(last) = last_token_time {
                                let interval = now.duration_since(last).as_secs_f64() * 1000.0;
                                chunk_intervals.push(interval);
                            }
                            last_token_time = Some(now);
                        }
                    }
                    line.clear();
                }
                Ok(Err(error)) => return Err(RequestError::StreamErr(error)),
                Err(_) => return Err(RequestError::Timeout),
            }
        }

        if let (Some(first), Some(last)) = (first_token_time, last_token_time) {
            let total_time = last.duration_since(first).as_secs_f64() * 1000.0;
            result.insert("total_time".to_string(), format!("{total_time:.3}"));
        }
        if !chunk_intervals.is_empty() {
            let max_interval = chunk_intervals.iter().copied().fold(f64::MIN, f64::max);
            let avg_interval = chunk_intervals.iter().sum::<f64>() / chunk_intervals.len() as f64;
            result.insert(
                "max_time_between_tokens".to_string(),
                format!("{max_interval:.3}"),
            );
            result.insert(
                "avg_time_between_tokens".to_string(),
                format!("{avg_interval:.3}"),
            );
        }
        Ok(result)
    }
}
