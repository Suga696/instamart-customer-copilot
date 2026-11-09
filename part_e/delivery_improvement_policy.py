"""Generate delivery recommendations from observed city/hour patterns."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PROC=ROOT/"data"/"processed"
OUT=PROC/"part_e"
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    city=pd.read_csv(PROC/"part_d"/"delivery_city_analysis.csv")
    hour=pd.read_csv(PROC/"part_d"/"delivery_hourly_analysis.csv")
    city["recommendation"]=city.delay_rate_pct.map(
        lambda x: "Review local fulfilment capacity and peak demand; run a measured operational test."
        if x>=city.delay_rate_pct.median() else
        "Maintain current operating practice and continue monitoring."
    )
    city.to_csv(OUT/"delivery_city_recommendations.csv",index=False)
    peak=hour.sort_values("avg_delivery_minutes",ascending=False).head(3)
    report={
        "city_recommendations":city.to_dict(orient="records"),
        "highest_average_delivery_hours":peak.to_dict(orient="records"),
        "suggested_actions":[
            "Review staffing and dispatch capacity during observed high-volume or high-delay periods.",
            "Monitor median and 90th-percentile delivery duration by city and hour.",
            "Track the share of orders exceeding the standard 30-minute promise.",
            "Compare before/after results before claiming an operational improvement."
        ],
        "limitations":[
            "No warehouse, zone, rider workload or weather variables are present.",
            "The 30-minute comparison is based on the supplied delivery policy; rain exceptions are not identifiable.",
            "Descriptive city/hour differences do not prove causes."
        ]
    }
    (OUT/"delivery_improvement_policy.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("Delivery improvement recommendations generated.")
if __name__=="__main__": main()
