import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROC=ROOT/"data"/"processed"
OUT=ROOT/"data"/"reports"
def load(rel):
    p=PROC/rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status":"Not generated yet"}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data={
        "data_quality":load("data_quality_report.json"),
        "customer":load("part_b/customer_intelligence_report.json"),
        "fraud":load("part_c/fraud_analytics_report.json"),
        "nlp":load("part_d/nlp_llm_report.json"),
        "business":load("part_e/business_policy_report.json"),
    }
    encoded=json.dumps(data,ensure_ascii=False).replace("</","<\\/")
    html="""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Instamart Customer Intelligence Dashboard</title>
    <style>
    body{font-family:Arial,sans-serif;margin:0;background:#f4f6f8;color:#17212b}
    header{background:#142d4e;color:white;padding:24px 5vw}
    main{max-width:1200px;margin:24px auto;padding:0 18px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
    .card{background:white;border:1px solid #dce3ea;border-radius:10px;padding:16px}
    .metric{font-size:28px;font-weight:bold;margin:8px 0}
    nav{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}
    button{padding:9px 13px;border:1px solid #9aabba;border-radius:6px;background:white;cursor:pointer}
    button.active{background:#142d4e;color:white}
    section{display:none}.visible{display:block}
    table{width:100%;border-collapse:collapse;background:white;margin-top:12px}
    th,td{text-align:left;padding:9px;border-bottom:1px solid #e4e9ee;font-size:14px}
    th{background:#eaf0f5}.muted{color:#596b7b;font-size:13px}
    .warning{background:#fff5df;padding:12px;border-left:4px solid #d49b25;margin:12px 0}
    </style></head><body>
    <header><h1>Instamart Customer Intelligence & Support Copilot</h1>
    <p>Integrated synthetic-data analytics | Five modules | Evidence-led decision support</p></header>
    <main><div class="warning">Synthetic educational dataset. Anomaly flags are not proof of fraud; human review is required.</div>
    <div id="metrics" class="grid"></div>
    <nav><button class="active" data-tab="quality">Data Quality</button>
    <button data-tab="customer">Customer Intelligence</button><button data-tab="fraud">Fraud Analytics</button>
    <button data-tab="nlp">NLP & Policy</button><button data-tab="business">Business Policies</button></nav>
    <section id="quality" class="visible"><h2>Data Quality Report</h2><div id="qualityContent"></div></section>
    <section id="customer"><h2>Customer Intelligence</h2><div id="customerContent"></div></section>
    <section id="fraud"><h2>Fraud Analytics</h2><div id="fraudContent"></div></section>
    <section id="nlp"><h2>NLP, Complaint Themes & Policy Retrieval</h2><div id="nlpContent"></div></section>
    <section id="business"><h2>Business Policy Recommendations</h2><div id="businessContent"></div></section>
    <p class="muted">Generated locally from project JSON reports. No external scripts, APIs or Python dashboard packages.</p></main>
    <script>
    const DATA=__DATA__;
    const el=(id,html)=>document.getElementById(id).innerHTML=html;
    const esc=x=>String(x??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
    function table(rows){if(!rows||!rows.length)return "<p class='muted'>No report data generated.</p>";let keys=Object.keys(rows[0]);return "<table><thead><tr>"+keys.map(k=>"<th>"+esc(k)+"</th>").join("")+"</tr></thead><tbody>"+rows.map(r=>"<tr>"+keys.map(k=>"<td>"+esc(typeof r[k]==="object"?JSON.stringify(r[k]):r[k])+"</td>").join("")+"</tr>").join("")+"</tbody></table>";}
    const dq=DATA.data_quality, ds=dq.datasets||{}, cust=DATA.customer, seg=cust.segmentation||{}, churn=cust.churn||{}, fraud=DATA.fraud.detection||{}, nlp=DATA.nlp.ticket_analysis||{};
    el("metrics",[["Customers",ds.customers?.cleaned?.row_count??"—"],["Cleaned orders",ds.orders?.cleaned?.row_count??"—"],["Customer clusters",seg.cluster_count??"—"],["Anomalies flagged",fraud.anomalies_flagged??"—"],["Support tickets",nlp.ticket_count??"—"]].map(x=>"<div class='card'><div class='muted'>"+esc(x[0])+"</div><div class='metric'>"+esc(x[1])+"</div></div>").join(""));
    el("qualityContent",table(Object.entries(ds).map(([name,v])=>({dataset:name,raw_rows:v.raw?.row_count,cleaned_rows:v.cleaned?.row_count,raw_missing:v.raw?.missing_values_total,cleaned_missing:v.cleaned?.missing_values_total,rows_removed:v.change_summary?.rows_removed}))));
    el("customerContent","<p>Clusters: "+esc(seg.cluster_count??"not available")+"</p><p>Churn selected model: "+esc(churn.selected_model_by_pr_auc??churn.status??"not available")+"</p>"+table(seg.cluster_profiles||[]));
    el("fraudContent","<p>Method: "+esc(fraud.method??"not available")+". Flagged records: "+esc(fraud.anomalies_flagged??"not available")+"</p><div class='warning'>"+esc(fraud.important_caveat??"Anomaly is not proof of fraud.")+"</div>");
    el("nlpContent","<p>"+esc(nlp.evidence_based_summary??"Complaint report not generated.")+"</p><pre>"+esc(JSON.stringify(DATA.nlp.policy_retrieval_evaluation||{},null,2))+"</pre>");
    el("businessContent","<pre>"+esc(JSON.stringify(DATA.business.recommendations||DATA.business,null,2))+"</pre>");
    document.querySelectorAll("nav button").forEach(b=>b.addEventListener("click",()=>{document.querySelectorAll("nav button").forEach(x=>x.classList.remove("active"));document.querySelectorAll("main section").forEach(x=>x.classList.remove("visible"));b.classList.add("active");document.getElementById(b.dataset.tab).classList.add("visible");}));
    </script></body></html>"""
    (OUT/"instamart_dashboard.html").write_text(html.replace("__DATA__",encoded),encoding="utf-8")
    (OUT/"dashboard_data.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
    print("Interactive dashboard created: data/reports/instamart_dashboard.html")
if __name__=="__main__": main()
