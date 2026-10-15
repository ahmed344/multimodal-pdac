/* Offline sample exploration: each FP64 column is loaded once on demand. */
'use strict';
(() => {
const meta=window.latentMetadata, mode=window.latentMode, cache=new Map(), pending=new Map();
const feature=document.getElementById('feature'), target=document.getElementById('target'), color=document.getElementById('color'), slide=document.getElementById('slide'), scale=document.getElementById('scale');
const plot=document.getElementById('plot'), status=document.getElementById('status');
function option(select,value,label=value){const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o);}
meta.features.forEach(v=>option(feature,v));meta.targets.forEach(v=>option(target,v));
Object.keys(meta.columns).filter(v=>!['row_id','umap_1','umap_2'].includes(v)).forEach(v=>option(color,v,v==='batch_id'?'Batch':v));
meta.batches.forEach(b=>option(slide,String(b.batch_id),b.batch));color.value='batch_id';
if(mode==='umap') ['feature-label','target-label','scale-label'].forEach(id=>document.getElementById(id).hidden=true);
window.latentAsset=(key,encoded)=>{
 const binary=atob(encoded),bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
 const view=new DataView(bytes.buffer),values=new Float64Array(bytes.length/8);for(let i=0;i<values.length;i++)values[i]=view.getFloat64(i*8,true);
 cache.set(key,values);const entry=pending.get(key);pending.delete(key);entry.resolve(values);entry.script.remove();
 // Retain at most twelve columns, not all 256 features after browsing.
 while(cache.size>12)cache.delete(cache.keys().next().value);
};
function loadColumn(name){
 const key=meta.columns[name].key;
 if(cache.has(key))return Promise.resolve(cache.get(key));if(pending.has(key))return pending.get(key).promise;
 let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});const script=document.createElement('script');
 pending.set(key,{promise,resolve,reject,script});script.src='assets/'+key+'.js';script.onerror=()=>{pending.delete(key);script.remove();reject(new Error('Cannot load '+script.src));};document.head.append(script);return promise;
}
function percentile(values,p){const sorted=Float64Array.from(values).sort();if(!sorted.length)return 1;const position=(sorted.length-1)*p,lo=Math.floor(position),hi=Math.ceil(position);return sorted[lo]+(sorted[hi]-sorted[lo])*(position-lo);}
const batchName=new Map(meta.batches.map(b=>[b.batch_id,b.batch]));
let generation=0;
async function draw(){
 const current=++generation;status.textContent='Loading saved columns…';window.latentViewer.ready=false;
 try{
  const names=mode==='umap'?['umap_1','umap_2']: [feature.value,target.value];
  const [x,originalY,colors,rows,batches]=await Promise.all([...names,color.value,'row_id','batch_id'].map(loadColumn));
  if(current!==generation)return;
  const y=Float64Array.from(originalY);
  if(mode==='scatter'&&scale.value==='logit')for(let i=0;i<y.length;i++){const d=y[i];y[i]=d>0?Math.log(Math.max(meta.epsilon,Math.min(1-meta.epsilon,d))/(1-Math.max(meta.epsilon,Math.min(1-meta.epsilon,d)))):NaN;}
  const indices=[];for(let i=0;i<x.length;i++)if(Number.isFinite(x[i])&&Number.isFinite(y[i])&&(slide.value==='all'||batches[i]===Number(slide.value)))indices.push(i);
  const finite=indices.filter(i=>Number.isFinite(colors[i])),missing=indices.filter(i=>!Number.isFinite(colors[i]));
  const categorical=color.value==='batch_id',signed=meta.columns[color.value].signed;
  const allFinite=Array.from(colors).filter(Number.isFinite),limit=percentile(allFinite.map(Math.abs),.99)||1;
  let marker={size:4,color:finite.map(i=>colors[i]),showscale:true,colorscale:meta.palettes[signed?'RdBu_r':'turbo'],autocolorscale:false,
   cmin:signed?-limit:percentile(allFinite,.01),cmax:signed?limit:percentile(allFinite,.99),colorbar:{title:{text:color.value},tickformat:'.8g'}};
  if(categorical){const palette=meta.batch_colors;marker.colorscale=palette.flatMap((c,i)=>[[i/palette.length,c],[(i+1)/palette.length,c]]);marker.reversescale=false;marker.cmin=-.5;marker.cmax=palette.length-.5;marker.colorbar={title:{text:'Batch'},tickvals:meta.batches.map(b=>b.batch_id),ticktext:meta.batches.map(b=>b.batch)};}
  const trace=(ids,style)=>({type:'scattergl',mode:'markers',x:ids.map(i=>x[i]),y:ids.map(i=>y[i]),marker:style,
   customdata:ids.map(i=>[rows[i],batches[i],colors[i]]),text:ids.map(i=>`Row ${formatNumber(rows[i],'integer')}<br>Slide ${batchName.get(batches[i])}<br>${names[0]}: ${formatNumber(x[i])}<br>${mode==='scatter'&&scale.value==='logit'?'logit('+names[1]+')':names[1]}: ${formatNumber(y[i])}<br>${color.value}: ${formatNumber(colors[i],categorical?'integer':'float')}`),hovertemplate:'%{text}<extra></extra>',showlegend:false});
  const traces=[];if(missing.length)traces.push(trace(missing,{size:4,color:'#ccc'}));traces.push(trace(finite,marker));
  await Plotly.react(plot,traces,{paper_bgcolor:'#fff',plot_bgcolor:'#fff',margin:{t:25},uirevision:'sample-view',xaxis:{title:{text:names[0]},tickformat:'.8g'},yaxis:{title:{text:names[1]+(mode==='scatter'&&scale.value==='logit'?' (logit)':'')},tickformat:'.8g'}},{responsive:true});
  formatTicks();plot.removeAllListeners('plotly_afterplot');plot.on('plotly_afterplot',formatTicks);
  status.textContent=`${formatNumber(indices.length,'integer')} of ${formatNumber(meta.count,'integer')} saved ${meta.split} samples · ${formatNumber(missing.length,'integer')} unavailable color values · colors clipped at displayed limits`;
  window.latentViewer.ready=true;window.latentViewer.shown=indices.length;
 }catch(error){status.textContent=error.message;console.error(error);}
}
function formatTicks(){plot.querySelectorAll('.xtick text,.ytick text,.cbaxis text').forEach(node=>{const raw=node.textContent.replaceAll('−','-');if(/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(raw))node.textContent=formatNumber(Number(raw));});}
[feature,target,color,slide,scale].forEach(select=>select.onchange=draw);
window.latentViewer={loadColumn,metadata:meta,ready:false,draw};draw();
})();
