import assert from 'node:assert/strict';
import { pickDisplaySeries } from './resourceLive.js';

const liveSeries = [{ step: 1, cpu: 10, ram: 40, procMem: 1.2, gpu: null }];
const logSeries = [{ step: 8, cpu: 33, ram: 50, procMem: 2.5, gpu: 22 }];

assert.deepEqual(pickDisplaySeries(logSeries, liveSeries), logSeries);
assert.deepEqual(pickDisplaySeries([], liveSeries), liveSeries);
assert.deepEqual(pickDisplaySeries(null, []), []);

console.log('PASS resource live fallback behavior');
