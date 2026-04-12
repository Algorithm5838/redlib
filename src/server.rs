#![allow(dead_code)]
#![allow(clippy::cmp_owned)]

use brotli::enc::{BrotliCompress, BrotliEncoderParams};
use bytes::Bytes;
use cached::proc_macro::cached;
use cookie::Cookie;
use core::f64;
use futures_lite::{future::Boxed, Future, FutureExt};
use http_body_util::{combinators::BoxBody, BodyExt, Empty, Full};
use std::convert::Infallible;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{header, HeaderMap, Method, Request, Response};
use hyper_util::rt::TokioIo;
use libflate::gzip;
use route_recognizer::{Params, Router};
use std::{
	cmp::Ordering,
	fmt::Display,
	io,
	pin::Pin,
	result::Result,
	str::{from_utf8, Split},
	string::ToString,
	sync::Arc,
};
use time::OffsetDateTime;
use tokio::net::TcpListener;

use crate::dbg_msg;

/// The unified body type used for all responses.
pub type Body = BoxBody<Bytes, Infallible>;

/// Create a response body from a string or bytes.
pub fn full<T: Into<Bytes>>(chunk: T) -> Body {
	Full::new(chunk.into()).boxed()
}

/// Create an empty response body.
pub fn empty() -> Body {
	Empty::<Bytes>::new().boxed()
}

const BANNED_USER_AGENTS: &[&str] = &[
	"AI2Bot",
	"Ai2Bot-Dolma",
	"Amazonbot",
	"Andibot",
	"Applebot",
	"Applebot-Extended",
	"Awario",
	"Brightbot 1.0",
	"Bytespider",
	"CCBot",
	"ChatGPT-User",
	"Claude-SearchBot",
	"Claude-User",
	"Claude-Web",
	"ClaudeBot",
	"Cotoyogi",
	"Crawlspace",
	"Datenbank Crawler",
	"Devin",
	"Diffbot",
	"DuckAssistBot",
	"Echobot Bot",
	"EchoboxBot",
	"FacebookBot",
	"Factset_spyderbot",
	"FirecrawlAgent",
	"FriendlyCrawler",
	"GPTBot",
	"Google-CloudVertexBot",
	"Google-Extended",
	"GoogleOther",
	"GoogleOther-Image",
	"GoogleOther-Video",
	"ICC-Crawler",
	"ISSCyberRiskCrawler",
	"ImagesiftBot",
	"Kangaroo Bot",
	"Meta-ExternalAgent",
	"Meta-ExternalFetcher",
	"MistralAI-User",
	"MistralAI-User/1.0",
	"MyCentralAIScraperBot",
	"NovaAct",
	"OAI-SearchBot",
	"Operator",
	"PanguBot",
	"Panscient",
	"Perplexity-User",
	"PerplexityBot",
	"PetalBot",
	"PhindBot",
	"Poseidon Research Crawler",
	"QualifiedBot",
	"QuillBot",
	"SBIntuitionsBot",
	"Scrapy",
	"SemrushBot",
	"SemrushBot-BA",
	"SemrushBot-CT",
	"SemrushBot-OCOB",
	"SemrushBot-SI",
	"SemrushBot-SWA",
	"Sidetrade indexer bot",
	"TikTokSpider",
	"Timpibot",
	"VelenPublicWebCrawler",
	"WARDBot",
	"Webzio-Extended",
	"YandexAdditional",
	"YandexAdditionalBot",
	"YouBot",
	"aiHitBot",
	"anthropic-ai",
	"bedrockbot",
	"cohere-ai",
	"cohere-training-data-crawler",
	"facebookexternalhit",
	"iaskspider/2.0",
	"img2dataset",
	"meta-externalagent",
	"meta-externalfetcher",
	"omgili",
	"omgilibot",
	"panscient.com",
	"quillbot.com",
	"wpbot",
];

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;
type BoxResponse = Pin<Box<dyn Future<Output = Result<Response<Body>, String>> + Send>>;

/// Compressors for the response Body, in ascending order of preference.
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum CompressionType {
	Passthrough,
	Gzip,
	Brotli,
}

/// All browsers support gzip, so if we are given `Accept-Encoding: *`, deliver
/// gzipped-content.
///
/// Brotli would be nice universally, but Safari (iOS, iPhone, macOS) reportedly
/// doesn't support it yet.
const DEFAULT_COMPRESSOR: CompressionType = CompressionType::Gzip;

impl CompressionType {
	/// Returns a `CompressionType` given a content coding
	/// in [RFC 7231](https://datatracker.ietf.org/doc/html/rfc7231#section-5.3.4)
	/// format.
	fn parse(s: &str) -> Option<Self> {
		let c = match s {
			// Compressors we support.
			"gzip" => Self::Gzip,
			"br" => Self::Brotli,

			// The wildcard means that we can choose whatever
			// compression we prefer. In this case, use the
			// default.
			"*" => DEFAULT_COMPRESSOR,

			// Compressor not supported.
			_ => return None,
		};

		Some(c)
	}
}

impl Display for CompressionType {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Gzip => write!(f, "gzip"),
			Self::Brotli => write!(f, "br"),
			Self::Passthrough => Ok(()),
		}
	}
}

pub struct Route<'a> {
	router: &'a mut Router<fn(Request<Body>) -> BoxResponse>,
	path: String,
}

pub struct Server {
	pub default_headers: HeaderMap,
	router: Router<fn(Request<Body>) -> BoxResponse>,
}

#[macro_export]
macro_rules! headers(
	{ $($key:expr => $value:expr),+ } => {
		{
			let mut m = hyper::HeaderMap::new();
			$(
				if let Ok(val) = hyper::header::HeaderValue::from_str($value) {
					m.insert($key, val);
				}
			)+
			m
		}
	 };
);

pub trait RequestExt {
	fn params(&self) -> Params;
	fn param(&self, name: &str) -> Option<String>;
	fn set_params(&mut self, params: Params) -> Option<Params>;
	fn cookies(&self) -> Vec<Cookie<'_>>;
	fn cookie(&self, name: &str) -> Option<Cookie<'_>>;
}

pub trait ResponseExt {
	fn cookies(&self) -> Vec<Cookie<'_>>;
	fn insert_cookie(&mut self, cookie: Cookie<'_>);
	fn remove_cookie(&mut self, name: String);
}

impl RequestExt for Request<Body> {
	fn params(&self) -> Params {
		self.extensions().get::<Params>().unwrap_or(&Params::new()).clone()
	}

	fn param(&self, name: &str) -> Option<String> {
		self.params().find(name).map(std::borrow::ToOwned::to_owned)
	}

	fn set_params(&mut self, params: Params) -> Option<Params> {
		self.extensions_mut().insert(params)
	}

	fn cookies(&self) -> Vec<Cookie<'_>> {
		self.headers().get("Cookie").map_or(Vec::new(), |header| {
			header
				.to_str()
				.unwrap_or_default()
				.split("; ")
				.map(|cookie| Cookie::parse(cookie).unwrap_or_else(|_| Cookie::from("")))
				.collect()
		})
	}

	fn cookie(&self, name: &str) -> Option<Cookie<'_>> {
		self.headers().get("Cookie").and_then(|header| {
			header
				.to_str()
				.unwrap_or_default()
				.split("; ")
				.find_map(|s| Cookie::parse(s).ok().filter(|c| c.name() == name))
		})
	}
}

impl ResponseExt for Response<Body> {
	fn cookies(&self) -> Vec<Cookie<'_>> {
		self.headers().get("Cookie").map_or(Vec::new(), |header| {
			header
				.to_str()
				.unwrap_or_default()
				.split("; ")
				.map(|cookie| Cookie::parse(cookie).unwrap_or_else(|_| Cookie::from("")))
				.collect()
		})
	}

	fn insert_cookie(&mut self, cookie: Cookie<'_>) {
		if let Ok(val) = header::HeaderValue::from_str(&cookie.to_string()) {
			self.headers_mut().append("Set-Cookie", val);
		}
	}

	fn remove_cookie(&mut self, name: String) {
		let removal_cookie = Cookie::build(name).path("/").http_only(true).expires(OffsetDateTime::now_utc());
		if let Ok(val) = header::HeaderValue::from_str(&removal_cookie.to_string()) {
			self.headers_mut().append("Set-Cookie", val);
		}
	}
}

impl Route<'_> {
	fn method(&mut self, method: &Method, dest: fn(Request<Body>) -> BoxResponse) -> &mut Self {
		self.router.add(&format!("/{}{}", method.as_str(), self.path), dest);
		self
	}

	/// Add an endpoint for `GET` requests
	pub fn get(&mut self, dest: fn(Request<Body>) -> BoxResponse) -> &mut Self {
		self.method(&Method::GET, dest)
	}

	/// Add an endpoint for `POST` requests
	pub fn post(&mut self, dest: fn(Request<Body>) -> BoxResponse) -> &mut Self {
		self.method(&Method::POST, dest)
	}
}

impl Default for Server {
	fn default() -> Self {
		Self::new()
	}
}

impl Server {
	pub fn new() -> Self {
		Self {
			default_headers: HeaderMap::new(),
			router: Router::new(),
		}
	}

	pub fn at(&mut self, path: &str) -> Route<'_> {
		Route {
			path: path.to_owned(),
			router: &mut self.router,
		}
	}

	pub fn listen(self, addr: &str) -> Boxed<Result<(), Box<dyn std::error::Error + Send + Sync>>> {
		let addr = addr.to_owned();
		let router = Arc::new(self.router);
		let default_headers = Arc::new(self.default_headers);

		async move {
			let address: std::net::SocketAddr = addr.parse().unwrap_or_else(|_| panic!("Cannot parse {addr} as address (example format: 0.0.0.0:8080)"));
			let listener = TcpListener::bind(address).await?;

			// Graceful shutdown signal
			let shutdown = async {
				#[cfg(windows)]
				tokio::signal::ctrl_c().await.expect("Failed to install CTRL+C signal handler");

				#[cfg(unix)]
				{
					let mut signal_terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("Failed to install SIGTERM signal handler");
					tokio::select! {
						_ = tokio::signal::ctrl_c() => (),
						_ = signal_terminate.recv() => ()
					}
				}
			};

			tokio::pin!(shutdown);

			loop {
				tokio::select! {
					Ok((stream, _)) = listener.accept() => {
						let io = TokioIo::new(stream);
						let router = Arc::clone(&router);
						let default_headers = Arc::clone(&default_headers);

						tokio::task::spawn(async move {
							let router: Arc<Router<fn(Request<Body>) -> BoxResponse>> = Arc::clone(&router);
							let svc = service_fn(move |req: Request<Incoming>| {
								let router = Arc::clone(&router);
								let default_headers = Arc::clone(&default_headers);

								async move {
									// Convert Incoming body to our Body type
									let (parts, incoming) = req.into_parts();
									let body_bytes = match incoming.collect().await {
										Ok(collected) => collected.to_bytes(),
										Err(_) => return Ok(Response::builder().status(500).body(empty()).unwrap()),
									};
									let req: Request<Body> = Request::from_parts(parts, full(body_bytes));

									let req_headers = req.headers().clone();
									let def_headers = (*default_headers).clone();

									// Catch robots.txt-disrespectful bots who still identify themselves
									if crate::utils::disable_indexing() {
										if let Some(user_agent) = req_headers.get("user-agent") {
											if let Ok(user_agent_str) = user_agent.to_str() {
												for banned in BANNED_USER_AGENTS {
													if user_agent_str.contains(banned) {
														return Ok(new_boilerplate(def_headers, req_headers, 403, full("Forbidden"))
															.await
															.unwrap_or_else(|_| Response::builder().status(403).body(empty()).unwrap()));
													}
												}
											}
										}
									}

									// Remove double slashes and decode encoded slashes
									let mut path = req.uri().path().replace("//", "/").replace("%2F", "/");

									// Remove trailing slashes
									if path != "/" && path.ends_with('/') {
										path.pop();
									}

									// Replace HEAD with GET for routing
									let (method, is_head) = match req.method() {
										&Method::HEAD => (&Method::GET, true),
										method => (method, false),
									};

									// Match the visited path with an added route
									let res: Result<Response<Body>, String> = match router.recognize(&format!("/{}{}", method.as_str(), path)) {
										Ok(found) => {
											let mut parammed = req;
											parammed.set_params(found.params().clone());

											let func = (found.handler().to_owned().to_owned())(parammed);
											match func.await {
												Ok(mut res) => {
													res.headers_mut().extend(def_headers.clone());
													if is_head {
														*res.body_mut() = empty();
													} else {
														let _ = compress_response(&req_headers, &mut res).await;
													}
													Ok(res)
												}
												Err(msg) => new_boilerplate(def_headers, req_headers, 500, if is_head { empty() } else { full(msg) }).await,
											}
										}
										Err(e) => new_boilerplate(def_headers, req_headers, 404, if is_head { empty() } else { full(e) }).await,
									};

									Ok::<Response<Body>, hyper::Error>(res.unwrap_or_else(|_| Response::builder().status(500).body(empty()).unwrap()))
								}
							});

							if let Err(_err) = hyper::server::conn::http1::Builder::new().serve_connection(io, svc).await {
								dbg_msg!("Error serving connection: {_err}");
							}
						});
					}
					_ = &mut shutdown => break,
				}
			}

			Ok(())
		}
		.boxed()
	}
}

/// Create a boilerplate Response for error conditions. This response will be
/// compressed if requested by client.
async fn new_boilerplate(
	default_headers: HeaderMap<header::HeaderValue>,
	req_headers: HeaderMap<header::HeaderValue>,
	status: u16,
	body: Body,
) -> Result<Response<Body>, String> {
	match Response::builder().status(status).body(body) {
		Ok(mut res) => {
			let _ = compress_response(&req_headers, &mut res).await;
			res.headers_mut().extend(default_headers.clone());
			Ok(res)
		}
		Err(msg) => Err(msg.to_string()),
	}
}

/// Determines the desired compressor based on the Accept-Encoding header.
///
/// This function will honor the [q-value](https://developer.mozilla.org/en-US/docs/Glossary/Quality_values)
///  for each compressor. The q-value is an optional parameter, a decimal value
/// on \[0..1\], to order the compressors by preference. An Accept-Encoding value
/// with no q-values is also accepted.
///
/// Here are [examples](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Accept-Encoding#examples)
/// of valid Accept-Encoding headers.
///
/// ```http
/// Accept-Encoding: gzip
/// Accept-Encoding: gzip, compress, br
/// Accept-Encoding: br;q=1.0, gzip;q=0.8, *;q=0.1
/// ```
#[cached]
fn determine_compressor(accept_encoding: String) -> Option<CompressionType> {
	if accept_encoding.is_empty() {
		return None;
	};

	struct CompressorCandidate {
		alg: CompressionType,
		q: f64,
	}

	impl Ord for CompressorCandidate {
		fn cmp(&self, other: &Self) -> Ordering {
			match self.q.total_cmp(&other.q) {
				Ordering::Equal => self.alg.cmp(&other.alg),
				ord => ord,
			}
		}
	}

	impl PartialOrd for CompressorCandidate {
		fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
			Some(self.cmp(other))
		}
	}

	impl PartialEq for CompressorCandidate {
		fn eq(&self, other: &Self) -> bool {
			(self.q == other.q) && (self.alg == other.alg)
		}
	}

	impl Eq for CompressorCandidate {}

	let mut cur_candidate = CompressorCandidate {
		alg: CompressionType::Passthrough,
		q: f64::NEG_INFINITY,
	};

	for val in accept_encoding.split(',') {
		let mut q: f64 = 1.0;
		let mut spl: Split<'_, char> = val.split(';');

		let compressor: CompressionType = match spl.next() {
			Some(s) => match CompressionType::parse(s.trim()) {
				Some(candidate) => candidate,
				None => continue,
			},
			None => continue,
		};

		if let Some(s) = spl.next() {
			if !(s.len() > 2 && s.starts_with("q=")) {
				return None;
			}

			match s[2..].parse::<f64>() {
				Ok(val) => {
					if (0.0..=1.0).contains(&val) {
						q = val;
					} else {
						return None;
					};
				}
				Err(_) => {
					return None;
				}
			}
		};

		let new_candidate = CompressorCandidate { alg: compressor, q };
		if let Some(ord) = new_candidate.partial_cmp(&cur_candidate) {
			if ord == Ordering::Greater {
				cur_candidate = new_candidate;
			}
		};
	}

	if cur_candidate.q == f64::NEG_INFINITY {
		None
	} else {
		Some(cur_candidate.alg)
	}
}

/// Compress the response body, if possible or desirable.
async fn compress_response(req_headers: &HeaderMap<header::HeaderValue>, res: &mut Response<Body>) -> Result<(), String> {
	// Check if the data is eligible for compression.
	if let Some(hdr) = res.headers().get(header::CONTENT_TYPE) {
		match from_utf8(hdr.as_bytes()) {
			Ok(val) => {
				let s = val.to_string();
				if !(s.starts_with("text/") || s.starts_with("application/json")) {
					return Ok(());
				};
			}
			Err(e) => {
				dbg_msg!(e);
				return Err(e.to_string());
			}
		};
	} else {
		return Ok(());
	};

	// Check the accept-encoding header
	let accept_encoding: String = match req_headers.get(header::ACCEPT_ENCODING) {
		None => return Ok(()),
		Some(hdr) => match String::from_utf8(hdr.as_bytes().into()) {
			Ok(val) => val,
			#[cfg(debug_assertions)]
			Err(e) => {
				dbg_msg!(e);
				return Ok(());
			}
			#[cfg(not(debug_assertions))]
			Err(_) => return Ok(()),
		},
	};

	let compressor: CompressionType = match determine_compressor(accept_encoding) {
		Some(c) => c,
		None => return Ok(()),
	};

	// Collect the body bytes
	let body_bytes: Vec<u8> = {
		// Swap body with empty, collect old body
		let old_body = std::mem::replace(res.body_mut(), empty());
		// Body error type is Infallible, so unwrap is safe
		old_body.collect().await.unwrap().to_bytes().to_vec()
	};

	// Don't bother compressing tiny responses
	if body_bytes.len() < 1452 {
		*res.body_mut() = full(body_bytes);
		return Ok(());
	}

	// Compress!
	match compress_body(compressor, body_bytes) {
		Ok(compressed) => {
			let headers = res.headers_mut();
			headers.insert(header::CONTENT_ENCODING, compressor.to_string().parse().unwrap());
			headers.remove(header::CONTENT_LENGTH);
			*res.body_mut() = full(compressed);
		}
		Err(e) => return Err(e),
	}

	Ok(())
}

/// Compresses a `Vec<u8>` given a [`CompressionType`].
// TTL of 600 (== 10 minutes) since compression is computationally expensive.
#[cached(size = 100, time = 600, result = true)]
fn compress_body(compressor: CompressionType, body_bytes: Vec<u8>) -> Result<Vec<u8>, String> {
	let mut reader = io::Cursor::new(body_bytes);

	let compressed: Vec<u8> = match compressor {
		CompressionType::Gzip => {
			let mut gz: gzip::Encoder<Vec<u8>> = match gzip::Encoder::new(Vec::new()) {
				Ok(gz) => gz,
				Err(e) => {
					dbg_msg!(e);
					return Err(e.to_string());
				}
			};

			match io::copy(&mut reader, &mut gz) {
				Ok(_) => match gz.finish().into_result() {
					Ok(compressed) => compressed,
					Err(e) => {
						dbg_msg!(e);
						return Err(e.to_string());
					}
				},
				Err(e) => {
					dbg_msg!(e);
					return Err(e.to_string());
				}
			}
		}

		CompressionType::Brotli => {
			let brotli_params = BrotliEncoderParams::default();
			let mut compressed = Vec::<u8>::new();
			match BrotliCompress(&mut reader, &mut compressed, &brotli_params) {
				Ok(_) => compressed,
				Err(e) => {
					dbg_msg!(e);
					return Err(e.to_string());
				}
			}
		}

		CompressionType::Passthrough => {
			return Err("unsupported compressor".to_string());
		}
	};

	Ok(compressed)
}

#[cfg(test)]
mod tests {
	use super::*;
	use brotli::Decompressor as BrotliDecompressor;
	use futures_lite::future::block_on;
	use lipsum::lipsum;
	use std::{boxed::Box, io};

	#[test]
	fn test_determine_compressor() {
		assert_eq!(determine_compressor("unsupported".to_string()), None);
		assert_eq!(determine_compressor("gzip".to_string()), Some(CompressionType::Gzip));
		assert_eq!(determine_compressor("*".to_string()), Some(DEFAULT_COMPRESSOR));

		assert_eq!(determine_compressor("gzip, br".to_string()), Some(CompressionType::Brotli));
		assert_eq!(determine_compressor("gzip;q=0.8, br;q=0.3".to_string()), Some(CompressionType::Gzip));
		assert_eq!(determine_compressor("br, gzip".to_string()), Some(CompressionType::Brotli));
		assert_eq!(determine_compressor("br;q=0.3, gzip;q=0.4".to_string()), Some(CompressionType::Gzip));

		assert_eq!(determine_compressor("gzip;q=NAN".to_string()), None);
	}

	#[test]
	fn test_compress_response() {
		macro_rules! ae_gen {
			($x:expr) => {
				$x.to_string().as_str()
			};

			($x:expr, $($y:expr),+) => {
				format!("{}, {}", $x.to_string(), ae_gen!($($y),+)).as_str()
			};
		}

		for accept_encoding in [
			"*",
			ae_gen!(CompressionType::Gzip),
			ae_gen!(CompressionType::Brotli, CompressionType::Gzip),
			ae_gen!(CompressionType::Brotli),
		] {
			let expected_encoding: CompressionType = match determine_compressor(accept_encoding.to_string()) {
				Some(s) => s,
				None => panic!("determine_compressor(accept_encoding.to_string()) => None"),
			};

			let mut req_headers = HeaderMap::new();
			req_headers.insert(header::ACCEPT_ENCODING, header::HeaderValue::from_str(accept_encoding).unwrap());

			let lorem_ipsum: String = lipsum(10000);
			let expected_lorem_ipsum = Vec::<u8>::from(lorem_ipsum.as_str());
			let mut res = Response::builder()
				.status(200)
				.header(header::CONTENT_TYPE, "text/plain")
				.body(full(lorem_ipsum))
				.unwrap();

			if let Err(e) = block_on(compress_response(&req_headers, &mut res)) {
				panic!("compress_response(&req_headers, &mut res) => Err(\"{e}\")");
			};

			assert_eq!(
				res
					.headers()
					.get(header::CONTENT_ENCODING)
					.unwrap_or_else(|| panic!("missing content-encoding header"))
					.to_str()
					.unwrap_or_else(|_| panic!("failed to convert Content-Encoding header::HeaderValue to String")),
				expected_encoding.to_string()
			);

			let body_vec = match block_on(res.into_body().collect()) {
				Ok(b) => b.to_bytes().to_vec(),
				Err(e) => panic!("{e}"),
			};

			if expected_encoding == CompressionType::Passthrough {
				assert!(body_vec.eq(&expected_lorem_ipsum));
				continue;
			}

			let mut body_cursor: io::Cursor<Vec<u8>> = io::Cursor::new(body_vec);

			let mut decoder: Box<dyn io::Read> = match expected_encoding {
				CompressionType::Gzip => match gzip::Decoder::new(&mut body_cursor) {
					Ok(dgz) => Box::new(dgz),
					Err(e) => panic!("{e}"),
				},
				CompressionType::Brotli => Box::new(BrotliDecompressor::new(body_cursor, expected_lorem_ipsum.len())),
				_ => panic!("no decompressor for {expected_encoding}"),
			};

			let mut decompressed = Vec::<u8>::new();
			if let Err(e) = io::copy(&mut decoder, &mut decompressed) {
				panic!("{e}");
			};

			assert!(decompressed.eq(&expected_lorem_ipsum));
		}
	}
}
