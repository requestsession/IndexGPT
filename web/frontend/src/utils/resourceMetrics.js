const RESOURCE_LINE_RE =
  /^\[Resource\]\s+step=(\d+)\s+\|\s+CPU\s+([0-9.]+|N\/A)%?\s+\|\s+RAM\s+([0-9.]+|N\/A)%?\s+\|\s+GPU\s+([0-9.]+|N\/A)%?(?:\s+\(([^)]*)\))?/;

function parseMaybeNumber(raw) {
  if (!raw || raw === 'N/A') return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

export function parseResourceLogs(logsText) {
  if (!logsText) return [];
  const lines = logsText.split(/\r?\n/);
  const points = [];

  for (const line of lines) {
    const matched = line.match(RESOURCE_LINE_RE);
    if (!matched) continue;
    points.push({
      step: Number(matched[1]),
      cpu: parseMaybeNumber(matched[2]),
      ram: parseMaybeNumber(matched[3]),
      gpu: parseMaybeNumber(matched[4]),
      gpuDetail: matched[5] || 'N/A'
    });
  }
  return points;
}

export function summarizeResourcePeaks(points) {
  const init = { cpu: null, ram: null, gpu: null };
  if (!points || !points.length) return init;

  return points.reduce((acc, point) => {
    const update = (k) => {
      if (point[k] == null) return;
      if (acc[k] == null || point[k] > acc[k]) acc[k] = point[k];
    };
    update('cpu');
    update('ram');
    update('gpu');
    return acc;
  }, init);
}
