"""Deterministic NER extractor for simple entity patterns."""

from __future__ import annotations

import json
import re

# Common city/country list (small, high-confidence subset)
_COMMON_LOCATIONS = {
    "berlin", "london", "paris", "tokyo", "beijing", "new york", "san francisco",
    "sydney", "moscow", "rome", "madrid", "toronto", "chicago", "boston",
    "seattle", "austin", "miami", "dubai", "singapore", "hong kong", "delhi",
    "mumbai", "bangalore", "são paulo", "rio", "cairo", "istanbul", "bangkok",
    "amsterdam", "barcelona", "munich", "frankfurt", "hamburg", "vienna",
    "zurich", "geneva", "prague", "warsaw", "budapest", "oslo", "stockholm",
    "copenhagen", "helsinki", "dublin", "edinburgh", "manchester", "liverpool",
    "glasgow", "cardiff", "belfast", "brisbane", "melbourne", "perth", "adelaide",
    "auckland", "wellington", "christchurch", "johannesburg", "cape town",
    "nairobi", "lagos", "casablanca", "tunis", "algiers", "tehran", "baghdad",
    "riyadh", "jeddah", "kuwait", "doha", "manama", "muscat", "abu dhabi",
    "kuala lumpur", "jakarta", "manila", "ho chi minh", "hanoi", "seoul",
    "busan", "taipei", "kaohsiung", "osaka", "kyoto", "nagoya", "fukuoka",
    "sapporo", "hiroshima", "kobe", "yokohama", "shanghai", "guangzhou",
    "shenzhen", "chengdu", "wuhan", "hangzhou", "nanjing", "tianjin",
    "xi'an", "chongqing", "harbin", "dalian", "qingdao", "xiamen",
    "suzhou", "wuxi", "changsha", "zhengzhou", "shijiazhuang", "taiyuan",
    "hefei", "nanchang", "kunming", "nanning", "lanzhou", "guiyang",
    "haikou", "yinchuan", "xining", "lhasa", "urumqi", "hohhot",
    "baotou", "ordos", "hulunbuir", "manzhouli", "mohe",
}

# Common date patterns
_DATE_PATTERNS = [
    re.compile(r"\b(?:last|next|this|every|in)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b", re.IGNORECASE),
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"),
    re.compile(r"\b(?:last|next|this)\s+(week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.IGNORECASE),
    re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.IGNORECASE),
]


def solve_ner(text: str) -> tuple[str, float] | None:
    """Extract simple named entities deterministically.

    Returns JSON array string and confidence, or None if extraction fails.
    """
    # Remove the instruction prefix to isolate the content.
    # Safely strip common prefixes like "Extract entities from:" or "Find NER in:".
    content = re.sub(
        r"^.*?(?:extract|find|get|label|identify).{0,30}?(?:from[:\s]+|in[:\s]+|about[:\s]+)\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
        count=1,
    )
    # If the aggressive strip removed too much (no remaining text), revert to full text.
    if not content.strip():
        content = text
    # Fallback: if no clear prefix, use text after the last colon.
    if content == text and ":" in text:
        content = text.rsplit(":", 1)[-1].strip()
    if not content.strip():
        content = text

    entities: list[dict[str, str]] = []

    # Person: Capitalized sequences of 2-3 words that aren't sentence starters
    person_matches = re.finditer(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
        content,
    )
    seen_spans: set[tuple[int, int]] = set()
    for m in person_matches:
        span = m.span()
        if span in seen_spans:
            continue
        name = m.group(1).strip()
        # Skip common non-person words
        if name.lower() in {"the", "a", "an", "this", "that", "it", "there"}:
            continue
        entities.append({"text": name, "type": "Person"})
        seen_spans.add(span)

    # Organization: look for capitalized words + tech/business suffixes
    org_matches = re.finditer(
        r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*?\s+(?:AI|Inc|Corp|Ltd|LLC|GmbH|Co|Company|Labs|Studio|Group|Foundation|Institute|University|College|School|Bank|Airlines|Systems|Technologies|Solutions|Media|Network|Partners|Associates|Consulting|Capital|Ventures|Fund|Trust|Union|Alliance|Exchange|Platform|Hub|Center|Centre|Agency|Bureau|Department|Office|Ministry|Council|Committee|Commission|Authority|Agency|Service|Administration|Organization|Organisation|NGO|UN|EU|NATO|WHO|WTO|IMF|World Bank|Google|Microsoft|Apple|Amazon|Meta|Netflix|Spotify|Uber|Airbnb|Slack|Salesforce|Oracle|IBM|Intel|AMD|NVIDIA|Qualcomm|Samsung|Sony|Huawei|Xiaomi|Alibaba|Tencent|Baidu|JD|Meituan|ByteDance|DJI|WeChat|WhatsApp|Instagram|Twitter|LinkedIn|YouTube|TikTok|Snapchat|Pinterest|Reddit|Twitch|Discord|Zoom|Teams|Dropbox|Box|Stripe|Square|PayPal|Venmo|Robinhood|Coinbase|Lyft|Doordash|Grubhub|Instacart|Shopify|Etsy|Ebay|Craigslist|Zillow|Redfin|Opendoor|Compass|Palantir|Snowflake|Databricks|Cockroach Labs|MongoDB|Elastic|Datadog|Splunk|New Relic|PagerDuty|Twilio|SendGrid|Mailchimp|HubSpot|Marketo|Pardot|Zendesk|Freshdesk|Intercom|Drift|Olark|LiveChat|Tidio|Crisp|Help Scout|Front|Gmail|Outlook|Yahoo|ProtonMail|Fastmail|Hey|Superhuman|Notion|Evernote|OneNote|Bear|Roam|Obsidian|Logseq|Craft|Apple Notes|Google Keep|Microsoft To Do|Todoist|Things|OmniFocus|Remember|Any\.do|TickTick|Habitica|Forest|Freedom|Cold Turkey|RescueTime|Toggl|Clockify|Harvest|Everhour|Hours|Timely|Memtime|Attentiv|Focusmate|Caveday|Flown|Sunsama|Akiflow|Motion|Reclaim|Clockwise|Amie|Routine|Notion Calendar|Cron|Cal\.com|Calendly|Doodle|When2meet|X.ai|Clara|Julie|Amy|Sidekick|Superpowered|Reclaim.ai|Clockwise|Motion|Amie|Routine|Notion Calendar|Cron|Cal.com|Calendly|Doodle|When2meet|X.ai|Clara|Julie|Amy|Sidekick|Superpowered|Reclaim.ai|Clockwise|Motion|Amie|Routine))\b",
        content,
    )
    # Actually that regex is too specific. Let's do a simpler approach.
    # Heuristic: any capitalized word sequence ending with AI, Labs, Inc, etc.
    org_simple = re.finditer(
        r"\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*?\s+(?:AI|Labs|Inc|Corp|Ltd|LLC|GmbH|Co\.?|Group|Studio|Network|Systems|Technologies|Solutions|Media|Partners|Associates|Consulting|Capital|Ventures|Fund|Trust|Union|Alliance|Exchange|Platform|Hub|Center|Agency|Bureau|Department|Office|Ministry|Council|Committee|Commission|Authority|Service|Administration|Organization|Organisation|Bank|Airlines|University|College|School))\b",
        content,
    )
    for m in org_simple:
        span = m.span()
        if span in seen_spans:
            continue
        org = m.group(1).strip()
        entities.append({"text": org, "type": "Organization"})
        seen_spans.add(span)

    # Also match known tech companies without suffix
    known_orgs = re.finditer(
        r"\b(Google|Microsoft|Apple|Amazon|Meta|Netflix|Spotify|Uber|Airbnb|Slack|Salesforce|Oracle|IBM|Intel|AMD|NVIDIA|Qualcomm|Samsung|Sony|Huawei|Xiaomi|Alibaba|Tencent|Baidu|ByteDance|DJI|WeChat|WhatsApp|Instagram|Twitter|LinkedIn|YouTube|TikTok|Snapchat|Pinterest|Reddit|Twitch|Discord|Zoom|Teams|Dropbox|Stripe|Square|PayPal|Robinhood|Coinbase|Lyft|Doordash|Grubhub|Instacart|Shopify|Etsy|Ebay|Craigslist|Zillow|Palantir|Snowflake|Databricks|MongoDB|Elastic|Datadog|Splunk|Twilio|Zendesk|HubSpot|Notion|Figma|Canva|Grammarly|OpenAI|Anthropic|Cohere|Mistral|Hugging Face|Stability AI|Midjourney|Runway|ElevenLabs|AssemblyAI|Deepgram|Speechmatics|Rev.ai|Voiceflow|Bland|Retell|PolyAI|Kore.ai|Avaamo|Rasa|Haptik|Yellow\.ai|Kata\.ai|Fireworks AI)\b",
        content,
    )
    for m in known_orgs:
        span = m.span()
        if span in seen_spans:
            continue
        org = m.group(1).strip()
        entities.append({"text": org, "type": "Organization"})
        seen_spans.add(span)

    # Location
    for loc in _COMMON_LOCATIONS:
        pattern = re.compile(rf"\b{re.escape(loc)}\b", re.IGNORECASE)
        for m in pattern.finditer(content):
            span = m.span()
            if span in seen_spans:
                continue
            entities.append({"text": m.group(0), "type": "Location"})
            seen_spans.add(span)

    # Date
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(content):
            span = m.span()
            if span in seen_spans:
                continue
            entities.append({"text": m.group(0), "type": "Date"})
            seen_spans.add(span)

    if not entities:
        return None

    return (json.dumps(entities, ensure_ascii=False), 0.85)
