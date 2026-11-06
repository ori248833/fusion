/* Synthetic events: test the actual report's interval calculations, not ROS. */
const assert = require('node:assert/strict');
const M = require('../my_launch/frame_metrics.js');
const events = [0, 0.5, 1, 1.5, 2, 4].map(t => [t, '0', 1, 'ok']);
assert.equal(M.windowStats(events, 0, 1).hz, null);
assert.equal(M.windowStats(events, 0.5, 1).partial, true);
assert.equal(M.windowStats(events, 0.5, 1).count, 2); // Startup includes t=0.
assert.equal(M.windowStats(events, 1.5, 1).count, 2); // (0.5, 1.5], excludes left edge.
assert.equal(M.windowStats(events, 1.5, 1).hz, 2);
assert.equal(M.windowStats(events, 3.5, 1).hz, 0);
assert.equal(M.windowStats(events, 3.5, 1).silence, 1);
assert.equal(M.windowStats(events, 3.5, 1).age, 1.5);
assert.equal(M.windowStats([], 3.5, 1).status, '尚未观测到输出');
assert.equal(M.windowStats([[5]], 3.5, 1).count, 0); // Future frames do not enter the window.
assert.equal(M.windowStats(events, 4, 1).silence, 1); // Boundary silence included.
assert.equal(M.gaps(events, 6, 3).threshold, 1.5);
assert.deepEqual(M.gaps(events, 6, 3).intervals.map(g => [g.start, g.end]), [[2,4],[4,6]]);
assert.deepEqual(M.gaps([], 6, 3).intervals, []);
assert.deepEqual(M.gaps([[1]], 6, 3).intervals, []); // Not enough intervals to infer a rate.
assert.equal(M.median([0.2,0.1,0.3,0.4]), 0.25);
console.log('Frame metrics: window boundaries, silence, startup, gaps and empty streams passed.');
