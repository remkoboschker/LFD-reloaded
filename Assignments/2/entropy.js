function calc() {
  const pPos = 3;
  const pNeg = 5;
  const c1Pos = 1;
  const c1Neg = 1;
  const c2Pos = 2;
  const c2Neg= 4;
  function l(cnt, tot){
    if(cnt === 0) {
      return 0;
    }
    return -(cnt / tot) * Math.log2(cnt / tot);
  }
  function ent(pos, neg) {

    const tot = pos + neg;
    return l(pos,tot) + l(neg,tot);
  }
  function chop(num) {
    return num.toString().slice(0,6);
  }
  const parent = ent(pPos, pNeg);
  const child1 = ent(c1Pos, c1Neg);
  const child2 = ent(c2Pos, c2Neg);
  const weightedAverage = (((c1Pos + c1Neg) / (pPos + pNeg)) * child1) + (((c2Pos + c2Neg) / (pPos + pNeg)) * child2);
  const gain = parent - weightedAverage;
  console.log(`${chop(parent)} & ${chop(child1)} & ${chop(child2)} & ${chop(weightedAverage)} & ${chop(gain)}`);
}

calc();