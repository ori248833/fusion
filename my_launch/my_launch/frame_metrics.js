/* Shared by the offline report and the standalone numerical checks. */
const FrameMetrics = (() => {
  function upperBound(events, time) {
    let lo = 0, hi = events.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (events[mid][0] <= time) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
  function lowerBound(events, time) {
    let lo = 0, hi = events.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (events[mid][0] < time) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
  function windowStats(events, time, width) {
    const start = Math.max(0, time - width);
    const begin = start === 0 ? lowerBound(events, 0) : upperBound(events, start);
    const end = upperBound(events, time);
    const duration = time - start;
    let previous = start, silence = 0;
    for (let i = begin; i < end; i++) {
      silence = Math.max(silence, events[i][0] - previous);
      previous = events[i][0];
    }
    silence = Math.max(silence, time - previous);
    return {
      start, begin, end, count: end - begin,
      hz: duration > 0 ? (end - begin) / duration : null,
      partial: duration < width,
      age: end ? Math.max(0, time - events[end - 1][0]) : null,
      silence,
      status: !end ? '尚未观测到输出' : (end === begin ? '窗口内无输出' : '有输出'),
    };
  }
  function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = sorted.length >>> 1;
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }
  function intervalStats(events, time, maxIntervals=100) {
    const end = upperBound(events, time), begin = Math.max(0, end-maxIntervals-1);
    const count = Math.max(0, end-begin-1);
    if (!count) return {hz:null, count:0, min:null, max:null, std:null};
    const deltas=[];
    for(let i=begin+1;i<end;i++)deltas.push(events[i][0]-events[i-1][0]);
    const mean=deltas.reduce((a,b)=>a+b,0)/count;
    const variance=deltas.reduce((sum,d)=>sum+(d-mean)**2,0)/count;
    return {hz:mean>0?1/mean:null, count, min:Math.min(...deltas),
      max:Math.max(...deltas), std:Math.sqrt(variance)};
  }
  function gaps(events, duration, multiplier) {
    const periods = events.slice(1).map((e, i) => e[0] - events[i][0]).filter(x => x > 0);
    const typical = median(periods);
    if (typical === null) return {typical, threshold: null, intervals: []};
    const threshold = typical * multiplier, intervals = [];
    for (let i = 1; i < events.length; i++) {
      const start = events[i - 1][0], end = events[i][0];
      if (end - start > threshold) intervals.push({start, end, kind: '帧间长间隔'});
    }
    if (duration - events[events.length - 1][0] > threshold) {
      intervals.push({start: events[events.length - 1][0], end: duration, kind: '尾部无输出（含停机阶段）'});
    }
    return {typical, threshold, intervals};
  }
  return {upperBound, lowerBound, windowStats, intervalStats, median, gaps};
})();
if (typeof module !== 'undefined') module.exports = FrameMetrics;
