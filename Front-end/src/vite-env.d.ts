/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Gốc API của backend Laravel, ví dụ http://127.0.0.1:8000/api */
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
