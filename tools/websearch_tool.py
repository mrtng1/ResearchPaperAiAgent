import xml.etree.ElementTree as ET
import requests
from datetime import datetime
from typing import Dict, List
import json

# ======================
# Web Search Tool (arXiv version)
# ======================
class ResearchPaperSearchTool:
    def search(self, topic: str, year: int, comparison: str, min_citations: int) -> str:
        try:
            results = self._search_arxiv(topic, year, comparison)
            return json.dumps(results)
        except Exception as e:
            print(f"Search error: {str(e)}")
            return json.dumps([])

    def _search_arxiv(self, query: str, year: int, comparison: str) -> List[Dict]:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 10,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)

        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            paper = self._parse_entry(entry, year, comparison)
            if paper:
                papers.append(paper)

        return papers[:5]

    def _parse_entry(self, entry, target_year: int, comparison: str) -> Dict:
        published = entry.find('{http://www.w3.org/2005/Atom}published').text
        paper_year = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ').year

        if comparison == "before" and paper_year >= target_year:
            return None
        if comparison == "after" and paper_year <= target_year:
            return None
        if comparison == "in" and paper_year != target_year:
            return None

        return {
            "title": entry.find('{http://www.w3.org/2005/Atom}title').text.strip(),
            "authors": [a.find('{http://www.w3.org/2005/Atom}name').text
                        for a in entry.findall('{http://www.w3.org/2005/Atom}author')],
            "year": paper_year,
            "link": entry.find('{http://www.w3.org/2005/Atom}id').text,
            "summary": entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
        }