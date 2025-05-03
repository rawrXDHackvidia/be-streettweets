import os
import csv
import json
import subprocess
import requests
import cloudinary
import cloudinary.uploader
import uuid
from fastapi import FastAPI, Body, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from io import BytesIO
from dotenv import load_dotenv
from datetime import datetime
from supabase import create_client, Client


# SETUP ---------------------------------------------------------------------
# load_dotenv('.env.production')
# load_dotenv('.env.development')
load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

cloudinary.config(
	cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
	api_key=os.getenv("CLOUDINARY_API_KEY"),
	api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

TWITTER_AUTH_TOKEN = os.getenv("TWITTER_AUTH_TOKEN")

# model for road damage detection
cv_model = YOLO('yolov8x-world.pt')
cv_model.set_classes(["sinkhole", "pothole", "crack"])

# model for NER and label mapping
NER_MODEL = "cahya/bert-base-indonesian-NER"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL, use_auth_token=HF_TOKEN)
model = AutoModelForTokenClassification.from_pretrained(NER_MODEL, use_auth_token=HF_TOKEN)
ner = pipeline(
	"ner",
	model=NER_MODEL,
	tokenizer=NER_MODEL,
	aggregation_strategy="simple",
)

# model for label_mapping
id2label = model.config.id2label

# UTILS ---------------------------------------------------------------------

def map_labels(results):
	if not results or not results[0]['entity_group'].startswith("LABEL_"):
		return results

	def get_label(entity_group):
		idx = int(entity_group.replace("LABEL_", ""))
		return id2label[idx]

	mapped = []
	for r in results:
		r_clean = r.copy()
		r_clean["entity_group"] = get_label(r["entity_group"])
		mapped.append(r_clean)
	return mapped

def run_tweet_harvest(search_keyword: str, limit: int, output_filename: str):
	try:
		command = [
			"npx", "-y", "tweet-harvest@2.6.1",
			"-o", output_filename,
			"-s", search_keyword,
			"--tab", "LATEST",
			"-l", str(limit),
			"--token", TWITTER_AUTH_TOKEN
		]
		result = subprocess.run(command, capture_output=True, text=True)

		if result.returncode != 0:
			return False, result.stderr

		return True, "Success"
	except Exception as e:
		return False, str(e)
# APIs ---------------------------------------------------------------------

@app.get('/')
def index():
	# return {'message': f'Hello, world! This is the {env["ENV"]} environment :3 !!!'}
	return {'message': f'Hello, world!'}

@app.post("/get-location")
async def get_location(text: str = Body(...)):
	raw_results = ner(text)
	readable_results = map_labels(raw_results)
	
	location_names = [
		r["word"]
		for r in readable_results
		if r["entity_group"] in ("LOC", "LOCATION")
	]
	 
	return {"location": location_names[0]}

@app.post("/detect-damage")
async def detect_damage(image_url: str = Body(...)):
	try:
		response = requests.get(image_url)
		response.raise_for_status()
		pil_image = Image.open(BytesIO(response.content)).convert("RGB")
	except Exception as e:
		return JSONResponse(status_code=400, content={"error": str(e)})

	results = cv_model(pil_image, conf=0.01)

	detections = []
	draw = ImageDraw.Draw(pil_image)

	for r in results:
		for box in r.boxes:
			cls = r.names[int(box.cls)]
			conf = float(box.conf)
			xyxy = box.xyxy.tolist()[0]  # [x1, y1, x2, y2]

			draw.rectangle(xyxy, outline="red", width=3)
			draw.text((xyxy[0], xyxy[1] - 10), f"{cls} {conf:.2f}", fill="red")

			detections.append({
				"label": cls,
				"confidence": conf,
				"bbox": xyxy
			})

	# save_path = "detected_image.png"
	# pil_image.save(save_path)
	
	if detections:
		top_prediction = detections[0]['label']

		damage_level = ''
			
		if top_prediction == 'crack':
			damage_level = 'light'
		if top_prediction == 'pothole':
			damage_level = 'medium'
		if top_prediction == 'sinkhole':
			damage_level = 'severe'
		
		return {"damage_level": damage_level}
	else:
		return {"message": "No damage detected."}

@app.post("/crawl_tweets")
def crawl_tweets(
	keyword: str = Query(..., description="Keyword Twitter, misal: #jalanrusak"),
	since: str = Query("2023-01-01"),
	until: str = Query("2025-04-30"),
	limit: int = Query(10)
):
	filename = "tweets-data/street_tweets.csv"
	query = f"{keyword} since:{since} until:{until}"

	success, msg = run_tweet_harvest(query, limit, filename)
	
	with open(filename, newline='', encoding='utf-8') as csvfile:
		reader = csv.DictReader(csvfile)
		data = list(reader)
	return JSONResponse(content=data, status_code=200)

@app.post("/poll-scrape")
async def poll_scrape():
	crawl_response = crawl_tweets('#jalanrusak')
	crawl_response = json.loads(crawl_response.body)
	text = crawl_response[0]['full_text']
	image_url = crawl_response[0]['image_url']
	username = crawl_response[0]['username']
	
	try:
		location_response = await get_location(text)
		location = location_response['location']

		damage_response = await detect_damage(image_url)	
		damage_level = damage_response['damage_level']

		report_id = str(uuid.uuid4())

		data = {
			"report_id": report_id,
			"created_at": str(datetime.now()),
			"username": username,
			"image_url": image_url,
			"location": location,
			"severity": damage_level,
			"status": 'validated'
		}
    	
		response = supabase.table("reports").insert(data).execute()

		return {
			"message": "Report scraped from X and submitted succesfully!",
			"report_id": report_id,
			"text": text,
			"image_url": image_url,
			"location": location,
			"damage_level": damage_level,
		}

	except: 
		return {
			"message": "Invalid report. Report must have location in text and photo of damaged road.",
			"text": text,
			"image_url": image_url,
		}

@app.post("/submit_report")
async def submit_report(
	location: str = Form(...),
	image: UploadFile = File(...)
):
	contents = await image.read()
	temp_filename = f"temp_{datetime.now().timestamp()}_{image.filename}"
	with open(temp_filename, "wb") as f:
		f.write(contents)

	image_url = ''
	try:
		upload_result = cloudinary.uploader.upload(temp_filename)
		image_url = upload_result.get("secure_url")
	finally:
		os.remove(temp_filename)

	try:
		damage_response = await detect_damage(image_url)	
		damage_level = damage_response['damage_level']
	except:
		return {
			"message": "Invalid report. Report must have a photo of the damaged road.",
			"image_url": image_url,
			"locataion:" : location,
		}

	report_id = str(uuid.uuid4())

	data = {
		"report_id": report_id,
		"created_at": str(datetime.now()),
		"username": None,
		"image_url": image_url,
		"location": location,
		"severity": damage_level,
		"status": 'validated'
	}
	
	response = supabase.table("reports").insert(data).execute()

	return JSONResponse({
		"message": "Report submitted successfully!",
		"report_id": report_id,
		"image_url": image_url,	
		"location": location,
		"damage_level": damage_level,
	})

@app.post('/get-all-reports')
def get_all_reports():
	response = supabase.table("reports").select("*").execute()
	data = response.data

	return JSONResponse(data)

@app.post('/get-report-by-id')
def get_report_by_id(report_id: str):

	response = supabase.table("reports").select("*").eq("report_id", report_id).execute()
	data = response.data

	if data:
		return JSONResponse(content=data[0], status_code=200)
	else:
		return JSONResponse(content={"message": "No matching reports found."}, status_code=404)