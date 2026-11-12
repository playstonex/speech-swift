# Changelog

## Unreleased

- Add per-load `DownloadOptions` and typed `DownloadRetryEvent` callbacks to
  download APIs and model loaders. Every failed attempt is logged and reported
  before its retry delay, including exhaustion. Existing retry defaults,
  progress callbacks, and complete-cache fallback are preserved. Task
  cancellation ends the retry ladder without emitting a retry event or wrapping
  cancellation as a download error. See
  [retry observation](shared-protocols.md#download-retry-observation).

- Preserve the final attempt's `DownloadError.stalled`, `checksumMismatch`,
  `invalidRemoteFileName`, and `networkUnavailable` cases and associated values
  after HuggingFace download retries. A stall previously became
  `networkUnavailable`; callers handling connectivity failures should now handle
  `stalled` too. Complete-cache fallback for stalls and other connectivity
  failures is unchanged. HTTP failures still use `failedToDownload` with retry
  context. See [download errors](shared-protocols.md#downloaderror).
