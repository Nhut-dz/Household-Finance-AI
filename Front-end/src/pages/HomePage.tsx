import { ArrowRight, LogIn } from 'lucide-react'
import type { PageKey } from '../data/profile'

/**
 * Decorative equalizer bars: hollow vertical bars of uneven height used as a
 * light background flourish. Heights are hard-coded rather than randomised so
 * the pattern stays stable across re-renders.
 */
const BAR_SETS = [
  [34, 58, 22, 70, 40, 84, 30, 52],
  [46, 26, 66, 38, 78, 30, 56],
  [62, 30, 48, 88, 36, 24, 54, 42],
  [28, 52, 36, 74, 44, 60],
]

function EqualizerBars({
  bars,
  className = '',
}: {
  bars: number[]
  className?: string
}) {
  return (
    <div
      aria-hidden
      className={`pointer-events-none flex items-end gap-1.5 ${className}`}
    >
      {bars.map((h, i) => (
        <span
          key={i}
          className="w-2 rounded-sm border border-brand-200"
          style={{ height: h }}
        />
      ))}
    </div>
  )
}

function NeedCard({
  image,
  title,
  desc,
  onClick,
}: {
  image: string
  title: string
  desc: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="group flex flex-col items-center rounded-3xl p-6 text-center transition hover:-translate-y-1"
    >
      <img
        src={image}
        alt=""
        className="mb-5 h-60 w-auto max-w-full object-contain sm:h-64"
      />
      <h3 className="text-lg font-bold text-slate-800">{title}</h3>
      <p className="mt-2 max-w-xs text-sm text-slate-500">{desc}</p>
    </button>
  )
}

export default function HomePage({
  onNavigate,
}: {
  onNavigate: (page: PageKey) => void
}) {
  return (
    <div className="space-y-16">
      {/* Hero */}
      {/* The illustration deliberately breaks out above the blue card, so this
          section keeps a top margin for the overflow and must not clip it. */}
      <section className="relative mt-16 rounded-[2rem] bg-brand-600 px-6 pb-8 pt-6 sm:mt-20 sm:px-12 sm:pb-10 sm:pt-8">
        <div className="relative z-10 flex flex-col gap-8 md:flex-row md:items-center md:gap-6">
          <div className="md:w-[60%]">
            <h1 className="text-4xl font-bold leading-[1.1] tracking-tight text-white sm:text-5xl lg:text-6xl">
              Tư vấn tài chính cho
              <br />
               hộ gia đình
            </h1>
            <div className="mt-8 flex flex-wrap gap-3">
              <button
                onClick={() => onNavigate('info')}
                className="inline-flex items-center gap-2 rounded-full bg-accent-400 py-2 pl-5 pr-2 text-sm font-semibold text-white shadow-sm transition hover:bg-accent-500"
              >
                Tư vấn ngay
                <span className="grid h-7 w-7 place-items-center rounded-full bg-white/25">
                  <ArrowRight size={15} />
                </span>
              </button>
              <button className="inline-flex items-center gap-2 rounded-full bg-accent-400 py-2 pl-5 pr-2 text-sm font-semibold text-white shadow-sm transition hover:bg-accent-500">
                Đăng nhập
                <span className="grid h-7 w-7 place-items-center rounded-full bg-white/25">
                  <LogIn size={15} />
                </span>
              </button>
            </div>
          </div>

          <div className="-mt-12 md:-mt-24 md:w-[40%] lg:-mt-28">
            <img
              src="/banner_icon.png"
              alt="Minh họa tăng trưởng tài chính hộ gia đình"
              className="mx-auto h-auto w-full max-w-sm md:max-w-none"
            />
          </div>
        </div>
      </section>

      {/* Needs */}
      <section className="relative">
        {/* Scattered equalizer bars: two along the top, two down each side */}
        <EqualizerBars
          bars={BAR_SETS[0]}
          className="absolute -left-2 top-16 hidden lg:flex"
        />
        <EqualizerBars
          bars={BAR_SETS[2]}
          className="absolute -right-2 top-28 hidden lg:flex"
        />
        <EqualizerBars
          bars={BAR_SETS[1]}
          className="absolute -top-4 left-[12%] hidden opacity-70 xl:flex"
        />
        <EqualizerBars
          bars={BAR_SETS[3]}
          className="absolute -top-2 right-[14%] hidden opacity-70 xl:flex"
        />
        <EqualizerBars
          bars={BAR_SETS[1]}
          className="absolute bottom-10 left-[4%] hidden opacity-60 xl:flex"
        />
        <EqualizerBars
          bars={BAR_SETS[0]}
          className="absolute bottom-16 right-[5%] hidden opacity-60 xl:flex"
        />

        <div className="text-center">
          <p className="text-lg font-semibold text-brand-700">Xin chào!</p>
          <h2 className="mt-1 text-3xl font-extrabold text-slate-800">
            Bạn đang có
            <br />
            nhu cầu?
          </h2>
        </div>

        <div className="mx-auto mt-10 grid max-w-3xl gap-6 sm:grid-cols-2">
          <NeedCard
            image="/img1.png"
            title="Tư vấn tài chính cho gia đình"
            desc="Tư vấn chi tiết tài chính cho gia đình bạn"
            onClick={() => onNavigate('info')}
          />
          <NeedCard
            image="/img2.png"
            title="Bạn cần tư vấn vay vốn ngân hàng?"
            desc="Tư vấn chi tiết lãi suất, kế hoạch trả nợ và hơn thế"
            onClick={() => onNavigate('info')}
          />
        </div>
      </section>
    </div>
  )
}
