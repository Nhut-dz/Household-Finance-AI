/**
 * Lớp gọi API dùng chung cho backend Laravel (Household-Finance-BE-App).
 *
 * Backend luôn trả về cùng một khung dữ liệu:
 *   thành công: { status: true,  message, result: { data } }
 *   thất bại:   { status: false, message, result: { errors: { field: [msg] } } }
 */

const BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000/api'
).replace(/\/+$/, '')

/** Lỗi validate theo từng field, đúng key mà backend trả về. */
export type FieldErrors = Record<string, string[]>

interface ApiEnvelope<T> {
  status?: boolean
  message?: string
  result?: { data?: T; errors?: FieldErrors }
}

export class ApiError extends Error {
  /** HTTP status; 0 khi không kết nối được máy chủ. */
  readonly status: number

  readonly fieldErrors: FieldErrors

  constructor(message: string, status: number, fieldErrors: FieldErrors = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.fieldErrors = fieldErrors
  }
}

const TOKEN_KEY = 'hf.authToken'

/** Bearer token của Sanctum, null khi đang dùng phiên khách. */
export const getAuthToken = () => window.localStorage.getItem(TOKEN_KEY)

export const setAuthToken = (token: string | null) =>
  token
    ? window.localStorage.setItem(TOKEN_KEY, token)
    : window.localStorage.removeItem(TOKEN_KEY)

async function request<T>(path: string, init: RequestInit): Promise<T> {
  const token = getAuthToken()

  let response: Response
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...init.headers,
      },
    })
  } catch {
    throw new ApiError(
      'Không kết nối được máy chủ. Vui lòng kiểm tra lại kết nối hoặc thử lại sau.',
      0,
    )
  }

  // 204 và các phản hồi rỗng khác không có body để đọc.
  const raw = await response.text()
  let body: ApiEnvelope<T> = {}
  if (raw) {
    try {
      body = JSON.parse(raw) as ApiEnvelope<T>
    } catch {
      throw new ApiError('Máy chủ trả về dữ liệu không hợp lệ.', response.status)
    }
  }

  if (!response.ok) {
    throw new ApiError(
      body.message ?? 'Đã có lỗi xảy ra, vui lòng thử lại.',
      response.status,
      body.result?.errors ?? {},
    )
  }

  return body.result?.data as T
}

export const apiGet = <T>(path: string) => request<T>(path, { method: 'GET' })

export const apiPost = <T>(path: string, payload: unknown) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(payload) })

export const apiPut = <T>(path: string, payload: unknown) =>
  request<T>(path, { method: 'PUT', body: JSON.stringify(payload) })

export const apiDelete = <T>(path: string) => request<T>(path, { method: 'DELETE' })
