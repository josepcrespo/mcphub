import { useEffect, useRef } from 'react';
import { useToast } from '@/contexts/ToastContext';
import { getApiUrl } from '@/utils/runtime';
import { getToken } from '@/utils/interceptors';

const EMBED_SYNC_ERROR_MARKER = '[EMBED_SYNC_ERROR]';
const RECONNECT_DELAY_MS = 5000;
const TOAST_DURATION_MS = 8000;
const TOAST_DEDUPE_WINDOW_MS = 30000;

const EmbeddingSyncAlertListener = () => {
  const { showToast } = useToast();
  const lastToastMessageRef = useRef<string>('');
  const lastToastAtRef = useRef<number>(0);

  useEffect(() => {
    let eventSource: EventSource | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let isMounted = true;

    const shouldSuppressToast = (message: string): boolean => {
      const now = Date.now();
      const isDuplicate =
        message === lastToastMessageRef.current && now - lastToastAtRef.current < TOAST_DEDUPE_WINDOW_MS;

      if (isDuplicate) {
        return true;
      }

      lastToastMessageRef.current = message;
      lastToastAtRef.current = now;
      return false;
    };

    const handleEmbeddingSyncError = (rawMessage: string) => {
      const cleanedMessage = rawMessage.replace(EMBED_SYNC_ERROR_MARKER, '').trim();
      const toastMessage =
        cleanedMessage ||
        'Embedding synchronization failed. Check smart routing logs and provider rate limits.';

      if (shouldSuppressToast(toastMessage)) {
        return;
      }

      showToast(toastMessage, 'error', TOAST_DURATION_MS);
    };

    const connect = () => {
      if (!isMounted) {
        return;
      }

      const token = getToken();
      if (!token) {
        reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
        return;
      }

      eventSource = new EventSource(getApiUrl(`/logs/stream?token=${token}`));

      eventSource.onmessage = (event) => {
        if (!isMounted) {
          return;
        }

        try {
          const data = JSON.parse(event.data);
          if (data?.type !== 'log') {
            return;
          }

          const message = String(data?.log?.message || '');
          if (!message.includes(EMBED_SYNC_ERROR_MARKER)) {
            return;
          }

          handleEmbeddingSyncError(message);
        } catch {
          // Ignore malformed stream messages.
        }
      };

      eventSource.onerror = () => {
        if (!isMounted) {
          return;
        }

        if (eventSource) {
          eventSource.close();
          eventSource = null;
        }

        reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
      };
    };

    connect();

    return () => {
      isMounted = false;
      if (eventSource) {
        eventSource.close();
      }
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
    };
  }, [showToast]);

  return null;
};

export default EmbeddingSyncAlertListener;
