---
layout: post
title: Analyzing Youtube comments of JD-Vance Munich speech
---

I watched the speech of JD-Vance on this [Youtube video](https://www.youtube.com/watch?v=VOc44fVvneI&list=WL&index=5) to the Munich Security Conference shared by Forbes. Here were the top comments shown on Youtube:
![_config.yml](/images/Screenshot-2025-02-17.png "Title")

They all appear to be biased toward JD Vance. A notable pattern appears in the comments section: numerous responses began with phrases like "As a European," "From the UK," "As a South African," and "From France." The repetitive nature of these geographical identifiers across different commentators raised questions about their authenticity.

To investigate this, I downloaded all comments using the YouTube API and classified them as positive, neutral, or negative regarding Vance's speech using Llama-3.3-70B. Looking at comments with likes showed a striking pattern:
* Comments with >1 like: 88.5% positive toward JD Vance speech
* Comments with >5 likes: 96.3% positive toward JD Vance speech
* Comments with >10 likes: 99.4% positive toward JD Vance speech

This **really** extreme skew in sentiment, particularly among highly-liked comments points toward large-scale manipulation of the comment section and coordinated engagement patterns. The phenomenon of YouTube comment manipulation is not new (for spam and politics) but the impact it can have on elections is worrisome.

PS: In case you are interested, feel free to try the [Code](https://github.com/geoalgo/jd-vance-comment-analysis/blob/main/JD-Vance-comment-analysis.ipynb) to generate the analysis (I also pushed the generated dataset) happy to get your feedback on the methodology.

