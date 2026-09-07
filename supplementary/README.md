# Supplementary Materials

This folder contains supplementary experimental analysis for **Beyond Static Temporal Structures: Discovering Temporal Periodicity for Sequential Recommendation** (ICDM 2026).

## Cold-Start Analysis

[View Supplementary PDF](RHYTHM_visible_history_analysis.pdf)

This supplementary analysis addresses the reviewers' concern regarding the robustness of RHYTHM under short interaction histories. In particular, sparse user histories may provide limited behavioral evidence for discovering personalized temporal contexts.

To investigate this issue, we evaluate RHYTHM across different visible-history-length groups on Ta-Feng, DHRD, and MegaMarket, comparing it with TCPSRec and TALE using NDCG@10 and HR@10. We also report the non-empty phase ratio to examine how the available temporal evidence varies with history length.

The results show that RHYTHM remains competitive in the shortest-history groups, while its relative advantage over strong temporal baselines becomes more pronounced as more behavioral history is available.
