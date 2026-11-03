
import React from 'react';
import {
  Activity,
  LayoutDashboard,
  Scan,
  ScanLine,
  Database,
  BarChart3,
  BookOpen,
  Settings,
  ChevronRight,
  CheckCircle2,
  AlertCircle,
  ArrowUpRight,
  Microscope,
  ShieldCheck,
  CircleHelp,
  FlaskConical,
} from 'lucide-react';

export type NavTab =
  | 'overview'
  | 'segmentation'
  | 'reliability'
  | 'dataset'
  | 'evaluation'
  | 'experiments'
  | 'documentation'
  | 'settings';

interface SidebarProps {
  currentTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  isBackendConnected: boolean;
}

const NAV_GROUPS: {
  title: string;
  items: {
    id: NavTab;
    label: string;
    icon: React.ElementType;
    badge?: string;
  }[];
}[] = [
  {
    title: 'Workspace',
    items: [
      { id: 'overview', label: 'Overview', icon: LayoutDashboard },
      { id: 'segmentation', label: 'Segmentation Studio', icon: Scan },
      {
        id: 'reliability',
        label: 'Segmentation Reliability Studio',
        icon: ScanLine,
      },
      {
        id: 'dataset',
        label: 'Dataset Explorer',
        icon: Database,
        badge: '780',
      },
    ],
  },
  {
    title: 'Research',
    items: [
      { id: 'evaluation', label: 'Model Evaluation', icon: BarChart3 },
      { id: 'experiments', label: 'Experiment History', icon: FlaskConical },
      { id: 'documentation', label: 'Documentation', icon: BookOpen },
    ],
  },
];

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onTabChange,
  isBackendConnected,
}) => {
  return (
    <aside className="flex h-screen min-h-0 w-[250px] shrink-0 flex-col overflow-hidden border-r border-slate-200 bg-[#f8fafb]">
      {/* Brand */}
      <div className="relative overflow-hidden bg-[#102f35] px-5 pb-6 pt-5">
        <div className="pointer-events-none absolute -right-8 -top-10 h-32 w-32 rounded-full border border-teal-400/10" />
        <div className="pointer-events-none absolute -right-2 -top-4 h-20 w-20 rounded-full border border-teal-400/10" />

        <div className="relative flex items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-teal-400 text-[#102f35] shadow-lg shadow-black/10">
            <Activity className="h-5 w-5" strokeWidth={2.5} />
          </div>

          <div className="min-w-0">
            <h1 className="text-[15px] font-bold tracking-tight text-white">
              MedSegment
              <span className="ml-1.5 text-teal-300">AI</span>
            </h1>
            <p className="mt-1 text-[10px] font-medium tracking-wide text-teal-100/60">
              MEDICAL IMAGING LAB
            </p>
          </div>
        </div>

        <div className="relative mt-5 flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.06] px-3 py-2.5">
          <Microscope className="h-4 w-4 shrink-0 text-teal-300" />
          <div className="min-w-0 flex-1">
            <p className="text-[11px] font-semibold text-white">
              Research workspace
            </p>
            <p className="mt-0.5 text-[10px] text-teal-100/55">
              Breast ultrasound
            </p>
          </div>
          <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400" />
        </div>
      </div>

      {/* Navigation */}
      <nav
        aria-label="Main navigation"
        className="min-h-0 flex-1 space-y-6 overflow-y-auto px-3 py-5"
      >
        {NAV_GROUPS.map((group) => (
          <div key={group.title}>
            <div className="mb-2 flex items-center justify-between px-3">
              <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400">
                {group.title}
              </span>
              <span className="text-[10px] tabular-nums text-slate-300">
                {String(group.items.length).padStart(2, '0')}
              </span>
            </div>

            <div className="space-y-1">
              {group.items.map((item) => {
                const Icon = item.icon;
                const isActive = currentTab === item.id;

                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => onTabChange(item.id)}
                    aria-current={isActive ? 'page' : undefined}
                    className={`group relative flex w-full items-center gap-3 overflow-hidden rounded-xl px-3 py-3 text-left transition-all duration-200 ${
                      isActive
                        ? 'bg-[#123f43] text-white shadow-md shadow-teal-950/10'
                        : 'text-slate-600 hover:bg-white hover:text-slate-900 hover:shadow-sm'
                    }`}
                  >
                    {isActive && (
                      <span className="absolute bottom-2.5 left-0 top-2.5 w-[3px] rounded-r-full bg-teal-300" />
                    )}

                    <span
                      className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-colors ${
                        isActive
                          ? 'bg-white/10 text-teal-200'
                          : 'bg-slate-100 text-slate-500 group-hover:bg-teal-50 group-hover:text-teal-700'
                      }`}
                    >
                      <Icon
                        className="h-[17px] w-[17px]"
                        strokeWidth={1.8}
                      />
                    </span>

                    <span className="min-w-0 flex-1 text-[12px] font-semibold leading-5">
                      {item.label}
                    </span>

                    {item.badge && (
                      <span
                        className={`rounded-md px-1.5 py-1 font-mono text-[10px] tabular-nums ${
                          isActive
                            ? 'bg-white/10 text-teal-100'
                            : 'bg-slate-100 text-slate-500 group-hover:bg-teal-50 group-hover:text-teal-700'
                        }`}
                      >
                        {item.badge}
                      </span>
                    )}

                    {isActive && (
                      <ChevronRight className="h-3.5 w-3.5 shrink-0 text-teal-200" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}

        {/* Configuration */}
        <div>
          <div className="mb-2 px-3 text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400">
            Configuration
          </div>

          <button
            type="button"
            onClick={() => onTabChange('settings')}
            aria-current={currentTab === 'settings' ? 'page' : undefined}
            className={`group flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left transition-all ${
              currentTab === 'settings'
                ? 'bg-[#123f43] text-white shadow-md shadow-teal-950/10'
                : 'text-slate-600 hover:bg-white hover:text-slate-900 hover:shadow-sm'
            }`}
          >
            <span
              className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
                currentTab === 'settings'
                  ? 'bg-white/10 text-teal-200'
                  : 'bg-slate-100 text-slate-500 group-hover:bg-teal-50 group-hover:text-teal-700'
              }`}
            >
              <Settings className="h-[17px] w-[17px]" strokeWidth={1.8} />
            </span>

            <span className="flex-1 text-[12px] font-semibold">Settings</span>

            <ArrowUpRight
              className={`h-3.5 w-3.5 ${
                currentTab === 'settings'
                  ? 'text-teal-200'
                  : 'text-slate-400 opacity-0 group-hover:opacity-100'
              }`}
            />
          </button>
        </div>
      </nav>

      {/* Backend connection card */}
      <div className="border-t border-slate-200 bg-white p-3">
        <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
          <div className="flex items-center justify-between gap-2 bg-slate-50 px-3 py-3">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-white text-slate-600 shadow-sm">
                <Activity className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-[11px] font-bold text-slate-800">
                  Inference engine
                </p>
                <p className="mt-0.5 text-[9px] text-slate-400">
                  FastAPI backend
                </p>
              </div>
            </div>

            <span
              className={`flex shrink-0 items-center gap-1.5 rounded-full px-2 py-1 text-[9px] font-bold ${
                isBackendConnected
                  ? 'bg-emerald-50 text-emerald-700'
                  : 'bg-amber-50 text-amber-700'
              }`}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  isBackendConnected ? 'bg-emerald-500' : 'bg-amber-500'
                }`}
              />
              {isBackendConnected ? 'Connected' : 'Offline'}
            </span>
          </div>

          <div className="p-3">
            <div className="flex items-start gap-2">
              {isBackendConnected ? (
                <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-600" />
              ) : (
                <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
              )}

              <p className="text-[10px] leading-relaxed text-slate-500">
                {isBackendConnected
                  ? 'Backend is reachable. Model availability depends on server health.'
                  : 'Cannot reach the backend. Check that the API server is running.'}
              </p>
            </div>

            <button
              type="button"
              onClick={() => onTabChange('settings')}
              className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white py-2 text-[10px] font-semibold text-slate-700 transition-colors hover:border-teal-200 hover:bg-teal-50 hover:text-teal-800"
            >
              <Settings className="h-3 w-3" />
              Backend settings
              <ChevronRight className="ml-auto h-3 w-3" />
            </button>
          </div>
        </div>

        {/* Research disclaimer */}
        <div className="mt-3 flex items-start gap-2 px-1 py-1">
          <ShieldCheck className="mt-0.5 h-3 w-3 shrink-0 text-slate-400" />
          <p className="text-[9px] leading-relaxed text-slate-400">
            For research and evaluation only. Not intended for clinical
            diagnosis.
          </p>
        </div>

        <div className="mt-1 flex items-center justify-between px-1 pb-1 pt-2">
          <span className="text-[9px] font-medium tracking-wide text-slate-400">
            MEDSEGMENT AI
          </span>
          <span className="flex items-center gap-1 text-[9px] text-slate-400">
            <CircleHelp className="h-3 w-3" />
            Research prototype
          </span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
