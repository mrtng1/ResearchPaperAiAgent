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
        """
        Searches for research papers.
        Note: The min_citations parameter is accepted, but the current arXiv API backend
        does not provide citation data, so results are not filtered by it.
        The 'citations' field in the results will reflect this.
        """
        try:
            results = self._search_arxiv(topic, year, comparison, min_citations_requested=min_citations)
            return json.dumps(results)
        except requests.exceptions.RequestException as re:
            print(f"HTTP Request error during search: {str(re)}")
            return json.dumps([{"error": "Failed to connect to arXiv API.", "details": str(re)}])
        except ET.ParseError as pe:
            print(f"XML Parsing error: {str(pe)}")
            return json.dumps([{"error": "Failed to parse arXiv API response.", "details": str(pe)}])
        except Exception as e:
            print(f"Generic search error: {str(e)}")
            return json.dumps([{"error": "An unexpected error occurred during search.", "details": str(e)}])

    def _search_arxiv(self, query: str, year: int, comparison: str, min_citations_requested: int) -> List[Dict]:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 10,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            paper_dict = self._parse_entry(entry, year, comparison, min_citations_requested)
            if paper_dict:
                papers.append(paper_dict)
        return papers[:10] # return top 10 papers

    def _parse_entry(self, entry, target_year: int, comparison: str, min_citations_requested: int) -> Dict:
        try:
            published_str = entry.findtext('{http://www.w3.org/2005/Atom}published')
            if not published_str:
                return None
            paper_year = datetime.strptime(published_str, '%Y-%m-%dT%H:%M:%SZ').year

            if comparison == "before" and paper_year >= target_year:
                return None
            if comparison == "after" and paper_year <= target_year:
                return None
            if comparison == "in" and paper_year != target_year:
                return None

            return {
                "title": entry.findtext('{http://www.w3.org/2005/Atom}title', 'N/A').strip(),
                "authors": [author.findtext('{http://www.w3.org/2005/Atom}name', 'N/A')
                            for author in entry.findall('{http://www.w3.org/2005/Atom}author')],
                "year": paper_year,
                "link": entry.findtext('{http://www.w3.org/2005/Atom}id', 'N/A'),
                "summary": entry.findtext('{http://www.w3.org/2005/Atom}summary', 'N/A').strip(),
                "citations": "N/A (arXiv API)"
            }
        except Exception as e:
            return None