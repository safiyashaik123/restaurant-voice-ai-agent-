export type Page = 'assistant' | 'menu' | 'reservations'
export type Status = 'idle' | 'listening' | 'processing' | 'speaking'
export interface ChatResponse {
  session_id: string
  text: string
  intent: string
  entities: Record<string, unknown>
  data: Record<string, unknown> | null
  success: boolean
  is_fallback: boolean
  needs_clarification: boolean
  missing_slots: string[]
}
export interface Message { id: string; role: 'assistant' | 'user'; text: string; audioUrl?: string; isError?: boolean }
export interface MenuItem { id: number; category: string; name: string; description: string | null; price: number; is_available: boolean; allergen_status: string }
export interface MenuResponse { categories: string[]; items: MenuItem[] }
export interface ReservationRequest { customer_name: string; customer_phone: string; party_size: number; reservation_date: string; reservation_time: string }
export interface ReservationSummary {
  reservation_id: number;
  reservation_date: string;
  reservation_time: string;
  party_size: number;
  status: string;
  table_number: string;
}
// export interface ReservationSummary { reservation_id: number; reservation_date: string; reservation_time: string; party_size: number; status: string }
export interface ReservationResponse extends ReservationRequest { access_code: string; reservation_id: number; table_number: string; table_capacity: number; table_id: number; duration_minutes: number; status: string; created_at: string }
