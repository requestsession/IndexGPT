import assert from 'node:assert/strict';
import { parseResourceLogs, summarizeResourcePeaks } from './resourceMetrics.js';

const logs = `
[Resource] step=1 | CPU 11.0% | RAM 50.0% | GPU N/A
[Resource] step=2 | CPU 21.0% | RAM 55.0% | GPU 40.0%
[Resource] step=3 | CPU N/A | RAM 57.0% | GPU 42.5%
`;

const points = parseResourceLogs(logs);
assert.equal(points.length, 3);
assert.equal(points[0].step, 1);
assert.equal(points[0].cpu, 11.0);
assert.equal(points[0].gpu, null);
assert.equal(points[2].cpu, null);
assert.equal(points[2].gpu, 42.5);

const peaks = summarizeResourcePeaks(points);
assert.equal(peaks.cpu, 21.0);
assert.equal(peaks.ram, 57.0);
assert.equal(peaks.gpu, 42.5);

console.log('PASS resource metrics parsing behavior');
