
import React, { useEffect, useState } from 'react';
import {
  Server,
  CheckCircle2,
  AlertCircle,
  LoaderCircle,
  Save,
  RefreshCw,
  Link2,
  Clock3,
  RotateCcw,
  Settings2,
} from 'lucide-react';

type ConnectionStatus = 'idle' | 'checking' | 'connected' | 'error';

const DEFAULT_API_URL = 'http://127.0.0.1:8000';

const STORAGE_KEY = 'medsegment_api_url';

export const SettingsView: React.FC = () => {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [savedApiUrl, setSavedApiUrl] = useState(DEFAULT_API_URL);
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>('idle');
  const [connectionMessage, setConnectionMessage] = useState(
    'Test the connection to check backend availability.'
  );
  const [lastChecked, setLastChecked] = useState<string | null>(null);
  const [savedSuccess, setSavedSuccess] = useState(false);

  useEffect(() => {
    try {
      const storedUrl = localStorage.getItem(STORAGE_KEY);

      if (storedUrl) {
        setApiUrl(storedUrl);
        setSavedApiUrl(storedUrl);
      }
    } catch {
      // Continue with the default URL if browser storage is unavailable.
    }
  }, []);

  const normalizeBaseUrl = (url: string) => {
    return url.trim().replace(/\/+$/, '').replace(/\/api\/health$/, '');
  };

  const handleTestConnection = async () => {
    const baseUrl = normalizeBaseUrl(apiUrl);

    if (!baseUrl) {
      setConnectionStatus('error');
      setConnectionMessage('Enter a valid backend URL.');
      return;
    }

    setConnectionStatus('checking');
    setConnectionMessage('Checking the FastAPI health endpoint...');
    setSavedSuccess(false);

    try {
      const response = await fetch(`${baseUrl}/api/health`);

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}.`);
      }

      const data: { status?: string } = await response.json();

      if (data.status !== 'ok') {
        throw new Error('The server responded, but its health status is not OK.');
      }

      setConnectionStatus('connected');
      setConnectionMessage('Backend is reachable and reporting healthy.');
      setLastChecked(new Date().toLocaleTimeString());
    } catch (error) {
      setConnectionStatus('error');

      setConnectionMessage(
        error instanceof Error
          ? `${error.message} Check the server URL, backend status, and CORS configuration.`
          : 'Unable to connect to the backend.'
      );
      setLastChecked(new Date().toLocaleTimeString());
    }
  };

  const handleSaveSettings = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const baseUrl = normalizeBaseUrl(apiUrl);

    if (!baseUrl || !/^https?:\/\//i.test(baseUrl)) {
      setConnectionStatus('error');
      setConnectionMessage('Enter a valid URL beginning with http:// or https://.');
      setSavedSuccess(false);
      return;
    }

    try {
      localStorage.setItem(STORAGE_KEY, baseUrl);
      setApiUrl(baseUrl);
      setSavedApiUrl(baseUrl);
      setSavedSuccess(true);
    } catch {
      setSavedSuccess(false);
      setConnectionMessage(
        'Could not save settings in browser storage. Check your browser privacy settings.'
      );
    }
  };

  const handleReset = () => {
    setApiUrl(DEFAULT_API_URL);
    setConnectionStatus('idle');
    setConnectionMessage('Test the connection to check backend availability.');
    setLastChecked(null);
    setSavedSuccess(false);
  };

  const isDirty = normalizeBaseUrl(apiUrl) !== savedApiUrl;

  return (
    <div className="mx-auto max-w-5xl space-y-6 pb-12">
      {/* Page heading */}
      <header className="border-b border-slate-200 pb-6">
        <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-teal-100 bg-teal-50 px-3 py-1.5 text-xs font-semibold text-teal-800">
          <Settings2 size={14} />
          Workspace configuration
        </div>

        <h1 className="text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
          Settings
        </h1>

        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
          Configure and verify the connection between MedSegment AI and your
          medical image segmentation backend.
        </p>
      </header>

      <form onSubmit={handleSaveSettings} className="space-y-5">
        {/* API configuration */}
        <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
          <div className="flex items-start gap-3 border-b border-slate-100 px-5 py-5 sm:px-6">
            <div className="rounded-xl bg-teal-50 p-3 text-teal-700">
              <Server size={20} />
            </div>

            <div className="min-w-0 flex-1">
              <h2 className="text-sm font-semibold text-slate-900">
                Backend connection
              </h2>
              <p className="mt-1 text-xs leading-5 text-slate-500">
                FastAPI service used for automatic prompt generation and
                MedSAM segmentation.
              </p>
            </div>

            <span className="hidden rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-medium text-slate-600 sm:inline-flex">
              REST API
            </span>
          </div>

          <div className="space-y-5 p-5 sm:p-6">
            <div className="space-y-2">
              <label
                htmlFor="backend-url"
                className="block text-xs font-semibold text-slate-700"
              >
                Backend base URL
              </label>

              <div className="flex flex-col gap-2 sm:flex-row">
                <div className="relative min-w-0 flex-1">
                  <Link2
                    size={16}
                    className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
                  />

                  <input
                    id="backend-url"
                    type="url"
                    required
                    value={apiUrl}
                    onChange={(event) => {
                      setApiUrl(event.target.value);
                      setSavedSuccess(false);
                      setConnectionStatus('idle');
                      setConnectionMessage(
                        'Test the connection to check backend availability.'
                      );
                    }}
                    placeholder={DEFAULT_API_URL}
                    className="w-full rounded-xl border border-slate-200 bg-slate-50 py-3 pl-10 pr-3 font-mono text-sm text-slate-800 outline-none transition focus:border-teal-500 focus:bg-white focus:ring-2 focus:ring-teal-100"
                  />
                </div>

                <button
                  type="button"
                  onClick={handleTestConnection}
                  disabled={connectionStatus === 'checking'}
                  className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs font-semibold text-slate-700 transition hover:border-teal-300 hover:bg-teal-50 hover:text-teal-800 disabled:cursor-wait disabled:opacity-60"
                >
                  {connectionStatus === 'checking' ? (
                    <LoaderCircle size={15} className="animate-spin" />
                  ) : (
                    <RefreshCw size={15} />
                  )}
                  {connectionStatus === 'checking' ? 'Checking...' : 'Test connection'}
                </button>
              </div>

              <p className="text-[11px] leading-5 text-slate-500">
                Health endpoint:{' '}
                <code className="break-all text-slate-700">
                  {normalizeBaseUrl(apiUrl) || DEFAULT_API_URL}/api/health
                </code>
              </p>
            </div>

            {/* Connection status */}
            <div
              role="status"
              aria-live="polite"
              className={`rounded-xl border p-4 ${
                connectionStatus === 'connected'
                  ? 'border-emerald-200 bg-emerald-50'
                  : connectionStatus === 'error'
                    ? 'border-rose-200 bg-rose-50'
                    : connectionStatus === 'checking'
                      ? 'border-amber-200 bg-amber-50'
                      : 'border-slate-200 bg-slate-50'
              }`}
            >
              <div className="flex items-start gap-3">
                {connectionStatus === 'connected' ? (
                  <CheckCircle2
                    size={18}
                    className="mt-0.5 shrink-0 text-emerald-600"
                  />
                ) : connectionStatus === 'error' ? (
                  <AlertCircle
                    size={18}
                    className="mt-0.5 shrink-0 text-rose-600"
                  />
                ) : connectionStatus === 'checking' ? (
                  <LoaderCircle
                    size={18}
                    className="mt-0.5 shrink-0 animate-spin text-amber-600"
                  />
                ) : (
                  <Server
                    size={18}
                    className="mt-0.5 shrink-0 text-slate-500"
                  />
                )}

                <div className="min-w-0 flex-1">
                  <p className="text-xs font-semibold text-slate-800">
                    {connectionStatus === 'connected'
                      ? 'Backend connected'
                      : connectionStatus === 'error'
                        ? 'Connection unsuccessful'
                        : connectionStatus === 'checking'
                          ? 'Testing connection'
                          : 'Connection not tested'}
                  </p>

                  <p className="mt-1 break-words text-xs leading-5 text-slate-600">
                    {connectionMessage}
                  </p>

                  {lastChecked && (
                    <p className="mt-2 flex items-center gap-1.5 text-[10px] text-slate-500">
                      <Clock3 size={12} />
                      Last checked at {lastChecked}
                    </p>
                  )}
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-slate-100 bg-white p-4">
              <p className="text-xs font-semibold text-slate-800">
                Required backend endpoints
              </p>

              <div className="mt-3 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <code className="text-xs text-slate-600">GET /api/health</code>
                  <span className="rounded-md bg-slate-100 px-2 py-1 text-[10px] text-slate-600">
                    Health check
                  </span>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-2">
                  <code className="text-xs text-slate-600">POST /api/segment</code>
                  <span className="rounded-md bg-slate-100 px-2 py-1 text-[10px] text-slate-600">
                    Image segmentation
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Save preferences */}
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">
                Configuration preferences
              </h2>
              <p className="mt-1 text-xs leading-5 text-slate-500">
                Save the backend URL in this browser for future sessions.
              </p>

              {savedSuccess && (
                <p className="mt-2 flex items-center gap-1.5 text-xs font-medium text-emerald-700">
                  <CheckCircle2 size={14} />
                  Settings saved successfully.
                </p>
              )}

              {!savedSuccess && isDirty && (
                <p className="mt-2 text-xs text-amber-700">
                  You have unsaved changes.
                </p>
              )}
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={handleReset}
                className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-200 px-4 py-2.5 text-xs font-semibold text-slate-600 transition hover:bg-slate-50"
              >
                <RotateCcw size={14} />
                Reset field
              </button>

              <button
                type="submit"
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-teal-700 px-5 py-2.5 text-xs font-semibold text-white shadow-sm transition hover:bg-teal-800 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-offset-2"
              >
                <Save size={15} />
                Save settings
              </button>
            </div>
          </div>
        </section>
      </form>

      <p className="px-1 text-[11px] leading-5 text-slate-400">
        Research software notice: connection status confirms API health only.
        It does not verify segmentation accuracy or clinical suitability.
      </p>
    </div>
  );
};
