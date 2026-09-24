# MANIFEST: what each file of this package supports

Every file is listed with the place in the manuscript or the Supplementary that reads it, and with
its grade. **hard** means a gate of the development repository, a figure generator or the manuscript
builder names the file itself; **soft** means an anchored script of this package reads it; **group**
means the file belongs to a directory named as a whole by the anchor in its row. `RUN_TO_TABLE.md`
gives the command behind each printed cell. Float numbers are the manuscript's own: Tables I to
VIII and Figs. 1 to 5 in the body, Tables S-1 to S-12 in the Supplementary.

| File | Grade | Supports |
|---|---|---|
| `.zenodo.json` | group | the package itself |
| `MANIFEST.md` | group | the package itself |
| `README.md` | group | the package itself |
| `RELEASE_NOTES.md` | group | the package itself |
| `RUN_TO_TABLE.md` | group | the package itself |
| `anaheim/anaheim_vi_all_zones.json` | group | Table IV, the attainable maximum |
| `anaheim/anaheim_vi_perimeter.json` | hard | check_internal_contradiction |
| `anaheim/anaheim_vi_perimeter_bandhull.json` | group | Table IV, the attainable maximum |
| `anaheim/eq7_route_sets_gamma0.95.json` | group | Table IV, the attainable maximum |
| `anaheim/eq7_route_sets_gamma0.99.json` | group | Table IV, the attainable maximum |
| `anaheim_arrival_ceiling.py` | group | the package itself |
| `anaheim_vi.py` | group | the package itself |
| `anaheim_vi_all_zones.py` | group | the package itself |
| `anaheim_vi_perimeter.py` | group | the package itself |
| `benchmark_network/outerface12_intervals.json` | group | Fig. 4 |
| `benchmark_network/outerface12_intervals.py` | group | Fig. 4 |
| `benchmark_network/plot_networks.py` | soft | Fig. 4 |
| `boundary_open_demo/arrival_sentinel_control.py` | group | the arrival-term sentinel control of Supplementary S-I |
| `boundary_open_demo/dqn.py` | group | the value-based learner every 5x5-grid and lattice cell trains |
| `boundary_open_demo/env.py` | group | the 5x5-grid environment of Fig. 1, Tables IV-V and Fig. 5 |
| `boundary_open_demo/env_boundary_dest.py` | group | the destination-on-a-peripheral-link variant of the grid environment, Tables IV-VI |
| `boundary_open_demo/exit_sentinel_control.py` | group | the exit-sentinel control of Supplementary S-I |
| `boundary_open_demo/four_cells_boundary_dest.py` | group | run_cell, the entry point of every 5x5-grid cell of Tables IV-V and Fig. 5 |
| `boundary_open_demo/four_cells_curves.py` | group | the generator of gform/four_cells_curves_8000_gform.json, Fig. 5 |
| `boundary_open_demo/interior_deep_control.py` | group | the interior-destination depth control of Supplementary S-I |
| `boundary_open_demo/interior_destination.py` | group | the interior-destination control of Supplementary S-I |
| `boundary_open_demo/sentinel_control.py` | group | the state-sentinel control of Supplementary S-I |
| `boundary_open_demo/sweep_extra.py` | group | the additional learner settings of Supplementary S-V |
| `boundary_open_demo/train.py` | group | the training loop the 5x5-grid solvers import |
| `boundary_open_demo/truncation_control.py` | group | the step-cap control of Supplementary S-I |
| `benchmark_network/benchmark_demo.py` | group | the Sioux Falls and Anaheim environment and learner, Tables IV-V |
| `boundary_open_demo/discount_condition.json` | group | Tables IV-V on the 5x5 grid; the generators of Figs. 4-5 |
| `boundary_open_demo/discount_condition.py` | group | Tables IV-V on the 5x5 grid; the generators of Figs. 4-5 |
| `boundary_open_demo/dqn_sensitivity.py` | group | Tables IV-V on the 5x5 grid; the generators of Figs. 4-5 |
| `boundary_open_demo/dqn_sensitivity_30seed.json` | group | Tables IV-V on the 5x5 grid; the generators of Figs. 4-5 |
| `boundary_open_demo/four_cells_curves.json` | soft | Tables IV-V on the 5x5 grid; the generators of Figs. 4-5 |
| `boundary_open_demo/matched_budget_grid.json` | group | Tables IV-V on the 5x5 grid; the generators of Figs. 4-5 |
| `boundary_open_demo/matched_budget_grid.py` | group | Tables IV-V on the 5x5 grid; the generators of Figs. 4-5 |
| `boundary_open_demo/optimal_vi_boundary_dest.py` | hard | build_integrated_docx |
| `boundary_open_demo/plot_experiment_design.py` | hard | build_integrated_docx, check_figure_generators |
| `boundary_open_demo/plot_reward_condition.py` | hard | check_figure_generators |
| `boundary_open_demo/table8_budget_intervals.json` | group | Tables IV-V on the 5x5 grid; the generators of Figs. 4-5 |
| `boundary_open_demo/table8_budget_intervals.py` | group | Tables IV-V on the 5x5 grid; the generators of Figs. 4-5 |
| `boundary_open_demo/term_optima_discounted.json` | group | Tables IV-V on the 5x5 grid; the generators of Figs. 4-5 |
| `boundary_open_demo/term_optima_discounted.py` | group | Tables IV-V on the 5x5 grid; the generators of Figs. 4-5 |
| `boundary_open_demo/travel_time_bdest.py` | hard | check_travel_time_cells |
| `boundary_open_demo/travel_time_bdest_ingrid.json` | hard | build_integrated_docx, check_table6_against_runs, check_travel_time_cells |
| `boundary_open_demo/travel_time_bdest_ingrid_8000.json` | hard | build_integrated_docx, check_table6_against_runs, check_travel_time_cells |
| `check_eval_sets.py` | group | the package itself |
| `eq7_route_sets.py` | group | the package itself |
| `eval_substrate/benchmark_network_adjudication.json` | group | Supplementary Table S-10 |
| `eval_substrate/benchmark_network_adjudication.py` | group | Supplementary Table S-10 |
| `eval_substrate/substrate.json` | hard | countcheck |
| `eval_substrate/substrate.py` | group | Supplementary Table S-10 |
| `exposure_ratio_real_networks.py` | group | the package itself |
| `fulltext_scan.py` | hard | fulltext_scan |
| `fulltext_scan/README.md` | group | Section III-C |
| `fulltext_scan/fulltext_scan_per_study.csv` | hard | fulltext_scan |
| `gates/check_figure_generators.py` | hard | check_figure_generators |
| `gates/countcheck.py` | hard | check_ceiling_table, countcheck, make_build, plot_family_merged |
| `gform/anaheim_arrival_term_gform.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/anaheim_arrival_term_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/anaheim_cell_gform.py` | group | module of `gform/anaheim_eval200_optimum_gform.py` |
| `gform/anaheim_eval200_optimum_gform.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/anaheim_eval200_optimum_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/anaheim_shaped_vi_gform.json` | hard | build_integrated_docx, check_internal_contradiction |
| `gform/anaheim_shaped_vi_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/anaheim_zone_geometry_gform.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/anaheim_zone_geometry_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/cross_condition_8000_gform.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/cross_condition_8000_gform.log` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/cross_condition_8000_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/eq7_route_sets_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/eq7_route_sets_gform_gamma0.90.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/eq7_route_sets_gform_gamma0.95.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/eq7_route_sets_gform_gamma0.99.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/eq7_route_sets_gform_gamma0.999.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/eq7_route_sets_gform_gamma0.9999.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/eq7_route_sets_gform_gamma1.0.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/eq7_stated_gammas.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/eq7_stated_gammas.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/eq7_threshold_gform.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/eq7_threshold_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/eq7_threshold_gform2.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/eq7_threshold_gform2.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/fig_reward_condition_8000_gform.png` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/four_cells_curves_8000_gform.json` | hard | build_integrated_docx, check_table6_against_runs |
| `gform/four_cells_curves_8000_gform.log` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/grid_gform.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/grid_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/grid_gform2.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/grid_gform2.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/link_rho_check.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/link_rho_check.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/optimal_vi_boundary_dest_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/optimal_vi_boundary_dest_gform2.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/plot_networks_gform.py` | hard | check_figure_generators |
| `gform/plot_reward_condition_8000_gform.py` | hard | check_figure_generators |
| `gform/retrain_released_common_gform.json` | hard | build_integrated_docx, check_eval30_optimum, check_evalset_ceilings, check_internal_contradiction |
| `gform/retrain_released_common_gform.log` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/retrain_released_common_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/sioux_cells_3000_gform.json` | hard | build_integrated_docx, check_table6_against_runs, check_travel_time_cells |
| `gform/sioux_cells_3000_gform.log` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/sioux_cells_3000_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/sioux_vi_gform.json` | hard | build_integrated_docx |
| `gform/sioux_vi_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/sioux_vi_tiecheck_gform.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/sioux_vi_tiecheck_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/term_optima_discounted_gform.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/term_optima_discounted_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/term_optima_discounted_gform2.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/term_optima_discounted_gform2.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/tie_paths_gform.json` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `gform/tie_paths_gform.py` | group | Tables IV-VI, Figs. 4-5, Eq. (7) |
| `m2c/cells/g8_c_closed_al3000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/g8_c_closed_al8000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/g8_c_open_al3000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/g8_c_open_al8000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/g8_p_closed_al3000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/g8_p_closed_al8000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/g8_p_open_al3000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/g8_p_open_al8000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/g8_p_open_tm3000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/g8_p_open_tm8000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/s5_c_closed_al3000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/s5_c_closed_al8000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/s5_c_open_al3000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/s5_c_open_al8000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/s5_p_closed_tm3000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/s5_p_open_tm3000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/s7_c_closed_al8000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/s7_c_open_al8000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/s7_p_closed_al8000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/s7_p_open_al8000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/s7_p_open_tm8000.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/si_c_closed_al8000_a.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/si_c_closed_al8000_b.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/si_c_open_al8000_a.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells/si_c_open_al8000_b.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells3/dqn_sensitivity_C.json` | group | Tables IV-V, the M2-C variant |
| `m2c/cells3/dqn_sensitivity_P.json` | group | Tables IV-V, the M2-C variant |
| `m2c/eq7_c.json` | group | Tables IV-V, the M2-C variant |
| `m2c/eq7_c.py` | group | Tables IV-V, the M2-C variant |
| `m2c/eq7_pairs_c.json` | group | Tables IV-V, the M2-C variant |
| `m2c/eq7_pairs_c.py` | group | Tables IV-V, the M2-C variant |
| `m2c/eq7_pairs_c_extra.json` | group | Tables IV-V, the M2-C variant |
| `m2c/run_archived.py` | group | Tables IV-V, the M2-C variant |
| `m2c/run_controls.py` | group | Tables IV-V, the M2-C variant |
| `m2c/selftest_shaping_c.py` | group | Tables IV-V, the M2-C variant |
| `m2c/shaping_m2c.py` | group | Tables IV-V, the M2-C variant |
| `m2c/step3_grid_c.py` | group | Tables IV-V, the M2-C variant |
| `m2c/table_v_m2c.json` | group | Tables IV-V, the M2-C variant |
| `m2c/train_m2c.py` | group | Tables IV-V, the M2-C variant |
| `m2c/train_m2c_g1.json` | hard | check_internal_contradiction |
| `m2c/train_m2c_g2.json` | group | Tables IV-V, the M2-C variant |
| `m2c/train_m2c_g3.json` | group | Tables IV-V, the M2-C variant |
| `m2c/train_m2c_g4.json` | group | Tables IV-V, the M2-C variant |
| `m2c/train_rest.py` | group | Tables IV-V, the M2-C variant |
| `m2c/train_rest2.py` | group | Tables IV-V, the M2-C variant |
| `m2c/train_rest_sioux.json` | group | Tables IV-V, the M2-C variant |
| `m2c/vi_m2c.py` | group | Tables IV-V, the M2-C variant |
| `networks/Anaheim_net.tntp` | soft | Fig. 4 and Table IV |
| `networks/Anaheim_trips.tntp` | group | Fig. 4 and Table IV |
| `networks/ChicagoSketch_net.tntp` | group | Fig. 4 and Table IV |
| `networks/SiouxFalls_net.tntp` | group | Fig. 4 and Table IV |
| `networks/SiouxFalls_node.tntp` | group | Fig. 4 and Table IV |
| `networks/anaheim_nodes.geojson` | soft | Fig. 4 and Table IV |
| `plot_family_merged.py` | hard | build_integrated_docx, check_appendix_consistency, check_figure_generators, plot_family_merged |
| `plot_family_reporting.py` | hard | check_figure_generators, check_internal_contradiction, plot_family_merged, plot_family_reporting |
| `related_reviews/README.md` | group | Supplementary Table S-6 and Section II-A |
| `related_reviews/adjacent_review_readthrough.json` | group | Supplementary Table S-6 and Section II-A |
| `related_reviews/adjacent_review_readthrough.py` | group | Supplementary Table S-6 and Section II-A |
| `related_reviews/crossreview_recall.json` | group | Supplementary Table S-6 and Section II-A |
| `related_reviews/crossreview_recall.py` | group | Supplementary Table S-6 and Section II-A |
| `related_reviews/overlap_refs.json` | group | Supplementary Table S-6 and Section II-A |
| `related_reviews/overlap_refs.py` | group | Supplementary Table S-6 and Section II-A |
| `related_reviews/overlap_refs_complete.json` | group | Supplementary Table S-6 and Section II-A |
| `related_reviews/overlap_refs_complete.py` | group | Supplementary Table S-6 and Section II-A |
| `related_reviews/reviewed_studies_refs.json` | group | Supplementary Table S-6 and Section II-A |
| `released_impl/arrival_reachable_eval_set.json` | hard | check_internal_contradiction |
| `released_impl/arrival_reachable_eval_set.py` | hard | check_internal_contradiction |
| `released_impl/arrival_reachable_eval_set_bandhull.json` | hard | check_internal_contradiction |
| `released_impl/baseline_arm.py` | group | Table VI and Supplementary S-I.D |
| `released_impl/baseline_arm_hull.json` | hard | check_internal_contradiction |
| `released_impl/baseline_vs_learned.py` | group | Table VI and Supplementary S-I.D |
| `released_impl/retrain_released.json` | group | Table VI and Supplementary S-I.D |
| `released_impl/retrain_released.py` | group | Table VI and Supplementary S-I.D |
| `released_impl/retrain_released_10seed.json` | hard | check_internal_contradiction |
| `released_impl/retrain_released_full.json` | group | Table VI and Supplementary S-I.D |
| `released_impl/retrain_released_hull_10seed.json` | hard | check_internal_contradiction |
| `reviewed_studies/ieee_xplore_export_records.csv` | hard | supplementary.md |
| `reviewed_studies_text/0.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/1.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/10.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/101.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/11.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/12.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/13.README.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/13.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/14.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/15.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/16.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/17.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/18.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/19.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/2.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/20.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/21.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/22.README.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/22.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/23.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/24.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/25.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/26.README.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/27.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/28.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/29.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/3.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/30.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/31.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/32.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/33.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/34.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/36.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/37.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/38.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/39.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/4.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/40.README.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/40.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/41.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/42.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/43.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/46.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/47.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/48.README.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/48.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/49.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/5.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/50.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/51.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/52.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/53.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/54.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/55.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/56.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/57.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/58.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/59.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/6.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/60.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/61.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/62.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/63.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/64.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/65.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/66.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/67.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/68.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/69.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/7.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/70.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/71.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/72.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/74.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/75.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/76.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/77.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/78.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/79.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/8.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/81.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/82.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/83.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/84.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/85.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/86.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/87.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/88.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/89.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/9.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/90.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/91.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/92.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/94.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/95.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/97.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/98.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `reviewed_studies_text/99.txt` | group | Table I and Supplementary Tables S-9, S-10; the input of the lexical scans |
| `screening_trail/assembly_stages.json` | hard | check_flow_stages |
| `screening_trail/assembly_stages.py` | hard | check_flow_stages |
| `screening_trail/plot_assembly.py` | hard | check_figure_generators, check_flow_stages |
| `search_rerun/exclusion_clause_records.csv` | hard | check_evidence_artifacts |
| `search_rerun/exclusion_clause_rerun.csv` | hard | check_evidence_artifacts |
| `search_rerun/exclusion_clause_rerun.json` | hard | check_evidence_artifacts |
| `search_rerun/exclusion_clause_rerun.py` | group | Supplementary S-IV.A and Table S-4 |
| `sioux30/boundary_set_diag.py` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed00.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed01.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed02.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed03.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed04.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed05.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed06.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed07.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed08.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed09.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed10.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed11.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed12.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed13.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed14.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed15.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed16.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed17.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed18.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed19.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed20.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed21.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed22.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed23.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed24.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed25.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed26.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed27.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed28.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/dest_boundary_seed29.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/run_cell.py` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `sioux30/summary_30seed.json` | group | Table V, the Sioux Falls rows at 8000 episodes |
| `study_record_claim_charting.csv` | group | the package itself |
| `study_record_reviewed_studies.csv` | hard | check_coverage_claims, check_internal_contradiction, countcheck, fulltext_scan, plot_family_merged |
| `study_record_simulator_audit_v6.csv` | hard | countcheck |
| `survey_axis_coverage.json` | group | the package itself |
| `survey_axis_coverage.py` | group | the package itself |
| `table6_rows.py` | group | the package itself |
