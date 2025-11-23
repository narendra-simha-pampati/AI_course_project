function selectTab(name){
  document.querySelectorAll('.tab').forEach(el=>el.classList.add('hidden'));
  document.getElementById(`tab-${name}`).classList.remove('hidden');
  document.querySelectorAll('.tab-btn').forEach(b=>{
    if(b.dataset.tab===name){ b.classList.add('bg-slate-700'); }
    else { b.classList.remove('bg-slate-700'); }
  });
}

document.querySelectorAll('.tab-btn').forEach(btn=>{
  btn.addEventListener('click', ()=> selectTab(btn.dataset.tab));
});
selectTab('caption');

// Captioning
const capFile = document.getElementById('cap-file');
const capPreview = document.getElementById('cap-preview');
capFile.addEventListener('change', ()=>{
  const f = capFile.files[0];
  if (f){
    const url = URL.createObjectURL(f);
    capPreview.src = url;
  }
});

document.getElementById('cap-run').addEventListener('click', async ()=>{
  const f = capFile.files[0];
  if(!f){ alert('Choose an image first'); return; }
  const fd = new FormData();
  fd.append('file', f);
  fd.append('max_new_tokens', document.getElementById('cap-max').value);
  fd.append('temperature', document.getElementById('cap-temp').value);
  fd.append('top_p', document.getElementById('cap-topp').value);
  fd.append('repetition_penalty', document.getElementById('cap-rep').value);
  fd.append('num_beams', document.getElementById('cap-beams').value);
  fd.append('prefix', document.getElementById('cap-prefix').value);
  fd.append('suffix', document.getElementById('cap-suffix').value);
  const out = document.getElementById('cap-output');
  out.textContent = 'Generating...';
  try{
    const res = await fetch('/api/caption', { method: 'POST', body: fd });
    const data = await res.json();
    if(data.error) throw new Error(data.error);
    out.textContent = data.caption;
  }catch(e){
    out.textContent = 'Error: ' + e.message;
  }
});

// Text to Image
const t2iStatus = document.getElementById('t2i-status');
document.getElementById('t2i-run').addEventListener('click', async ()=>{
  const payload = {
    prompt: document.getElementById('t2i-prompt').value,
    negative_prompt: document.getElementById('t2i-neg').value,
    style: document.getElementById('t2i-style').value,
    steps: document.getElementById('t2i-steps').value,
    guidance: document.getElementById('t2i-guidance').value,
    width: document.getElementById('t2i-width').value,
    height: document.getElementById('t2i-height').value,
    seed: document.getElementById('t2i-seed').value,
  };
  if(!payload.prompt){ alert('Enter a prompt'); return; }
  t2iStatus.textContent = 'Generating image... first run may take a while.';
  try{
    const res = await fetch('/api/txt2img', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await res.json();
    if(data.error) throw new Error(data.error);
    document.getElementById('t2i-output').src = 'data:image/png;base64,' + data.image_base64;
    t2iStatus.textContent = `Done (seed: ${data.seed})`;
  }catch(e){
    t2iStatus.textContent = 'Error: ' + e.message;
  }
});

// Summarize
const sumOut = document.getElementById('sum-output');
document.getElementById('sum-run').addEventListener('click', async ()=>{
  sumOut.textContent = 'Summarizing...';
  const payload = {
    text: document.getElementById('sum-text').value,
    min_length: document.getElementById('sum-min').value,
    max_length: document.getElementById('sum-max').value,
  };
  try{
    const res = await fetch('/api/summarize', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await res.json();
    if(data.error) throw new Error(data.error);
    sumOut.textContent = data.summary;
  }catch(e){
    sumOut.textContent = 'Error: ' + e.message;
  }
});

// Elaborate
const elabOut = document.getElementById('elab-output');
document.getElementById('elab-run').addEventListener('click', async ()=>{
  elabOut.textContent = 'Elaborating...';
  const payload = {
    text: document.getElementById('elab-text').value,
    tone: document.getElementById('elab-tone').value,
    length: document.getElementById('elab-length').value,
    creativity: document.getElementById('elab-creativity').value,
  };
  try{
    const res = await fetch('/api/elaborate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await res.json();
    if(data.error) throw new Error(data.error);
    elabOut.textContent = data.elaboration;
  }catch(e){
    elabOut.textContent = 'Error: ' + e.message;
  }
});
