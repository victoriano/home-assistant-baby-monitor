export function formatTrendDate(dateKey: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateKey);
  if (!match) return dateKey;
  return `${match[3]}/${match[2]}`;
}

export function medianMinutes(values: readonly number[]): number {
  const sorted = values
    .filter((value) => Number.isFinite(value))
    .map((value) => Math.max(0, value))
    .sort((left, right) => left - right);
  if (!sorted.length) return 0;
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2) return Math.round(sorted[middle]);
  return Math.round((sorted[middle - 1] + sorted[middle]) / 2);
}
