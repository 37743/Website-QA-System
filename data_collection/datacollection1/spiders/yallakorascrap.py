import scrapy
from datetime import datetime, timedelta


class YallakoraSpider(scrapy.Spider):
    name = "yallakorascrap"
    allowed_domains = ["yallakora.com"]

    custom_settings = {
        "FEED_EXPORT_ENCODING": "utf-8",
        "ROBOTSTXT_OBEY": False,
        "DOWNLOAD_DELAY": 0.5,
    }

    def start_requests(self):
        start_date_str = getattr(self, 'start_date', "2026-03-05")
        end_date_str = getattr(self, 'end_date', "2026-03-19")
        
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        current_date = start_date
        while current_date <= end_date:
            formatted = f"{current_date.month}/{current_date.day}/{current_date.year}"
            url = f"https://www.yallakora.com/match-center?date={formatted}#days"

            yield scrapy.Request(
                url=url,
                callback=self.parse_match_center,
                meta={"match_date": current_date.strftime("%Y-%m-%d")}
            )
            current_date += timedelta(days=1)

    # --------------------------------------------------
    # Parse the daily match-center page and extract match links
    # --------------------------------------------------
    def parse_match_center(self, response):
        match_date = response.meta.get("match_date")

        # collect only real match links
        links = response.css('a[href*="/match/"]::attr(href)').getall()

        seen = set()
        for link in links:
            full_url = response.urljoin(link)

            # keep only unique links containing /match/<id>/
            if "/match/" in full_url and full_url not in seen:
                seen.add(full_url)
                yield scrapy.Request(
                    url=full_url,
                    callback=self.parse_match_details,
                    meta={"match_date": match_date}
                )

    # --------------------------------------------------
    # Parse each match page
    # --------------------------------------------------
    def parse_match_details(self, response):
        match_date = response.meta.get("match_date")

        # ===== Main result block only =====
        result_block = response.css("section.mtchDtlsRslt")
        score_block = result_block.css("div.matchScoreInfo")

        # team names from the score header only
        team_a = self.clean_text(
            score_block.css("div.team.teamA > a > p").xpath("string(.)").get()
        )
        team_b = self.clean_text(
            score_block.css("div.team.teamB > a > p").xpath("string(.)").get()
        )

        # remove winner icon side effects if any weird spacing appears
        team_a = team_a.replace(" ,", "").strip()
        team_b = team_b.replace(" ,", "").strip()

        # scores
        score_a = self.clean_text(score_block.css("div.result span.a::text").get())
        score_b = self.clean_text(score_block.css("div.result span.b::text").get())

        # status, date, time
        status = self.clean_text(result_block.css("p.status::text").get())
        page_date = self.clean_text(result_block.css(".matchDateInfo .date::text").get())
        page_time = self.clean_text(result_block.css(".matchDateInfo .time::text").get())

        # competition + round
        competition = self.clean_text(
            result_block.css(".tourNameBtn > a:first-child::text").get()
        )

        round_name = self.clean_text(
            result_block.css(".tourNameBtn > a:nth-child(2) p").xpath("string(.)").get()
        )

        # ===== Scorers from scorer block only =====
        scorers = {
            "team_a": [],
            "team_b": []
        }

        scorer_block = response.css("div.matcResultAction.scorer")

        for s in scorer_block.css("div.team.teamA.playerScorers div.goal"):
            player_name = self.clean_text(s.css("span.playerName::text").get())
            goal_time = self.clean_text(s.css("span.time::text").get())

            if player_name or goal_time:
                scorers["team_a"].append({
                    "player_name": player_name,
                    "goal_time": goal_time
                })

        for s in scorer_block.css("div.team.teamB.playerScorers div.goal"):
            player_name = self.clean_text(s.css("span.playerName::text").get())
            goal_time = self.clean_text(s.css("span.time::text").get())

            if player_name or goal_time:
                scorers["team_b"].append({
                    "player_name": player_name,
                    "goal_time": goal_time
                })

        # ===== Stats from #stats only =====
        stats = {}

        for li in response.css("div#stats div.statsDiv ul > li"):
            stat_name = self.clean_text(li.css("div.desc::text").get())
            stat_team_a = self.clean_text(li.css("div.team.teamA::text").get())
            stat_team_b = self.clean_text(li.css("div.team.teamB::text").get())

            if stat_name:
                stats[stat_name] = {
                    "team_a": stat_team_a,
                    "team_b": stat_team_b
                }

        # ===== Events from #events only =====
        events = []

        for ev in response.css("div#events div.timeline.events ul > li"):
            classes = ev.attrib.get("class", "")
            minute = self.clean_text(ev.css("div.min::text").get())
            description = self.clean_text(ev.css("p.description").xpath("string(.)").get())

            # skip empty referee separator rows if wanted
            if minute or description:
                events.append({
                    "class": classes,
                    "minute": minute,
                    "description": description
                })

        yield {
            "url": response.url,
            "match_date_input": match_date,
            "page_date": page_date,
            "page_time": page_time,
            "competition": competition,
            "round": round_name,
            "status": status,

            "team_a": team_a,
            "team_b": team_b,
            "score_a": score_a,
            "score_b": score_b,

            "scorers": scorers,
            "stats": stats,
            "events": events,
        }

    def clean_text(self, value):
        if value is None:
            return None
        return " ".join(value.split()).strip()