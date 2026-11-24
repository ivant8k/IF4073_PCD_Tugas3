classdef SegmentationApp < matlab.apps.AppBase
    % SEGMENTATIONAPP Aplikasi GUI Terpadu Tugas 3 IF4073 Pemrosesan Citra Digital.
    %
    % Mengintegrasikan tiga pendekatan segmentasi citra dalam satu antarmuka 3 Tab:
    %   Tab 1: Segmentasi Citra dengan Deteksi Tepi
    %   Tab 2: Segmentasi Klasik dengan Split and Merge
    %   Tab 3: Semantic Segmentation dengan Deep Learning (CamVid 11 Kelas)

    properties (Access = public)
        UIFigure            matlab.ui.Figure
        MainGrid            matlab.ui.container.GridLayout
        HeaderPanel         matlab.ui.container.Panel
        AppTitleLabel       matlab.ui.control.Label
        AppSubtitleLabel    matlab.ui.control.Label
        TabGroup            matlab.ui.container.TabGroup

        % --- Tab 1: Deteksi Tepi & Segmentasi Objek ---
        EdgeTab             matlab.ui.container.Tab
        EdgeTabGrid         matlab.ui.container.GridLayout
        EdgeCtrlPanel       matlab.ui.container.Panel
        EdgeCtrlGrid        matlab.ui.container.GridLayout
        EdgeLoadBtn         matlab.ui.control.Button
        EdgeFileLabel       matlab.ui.control.Label
        EdgeOpDropDown      matlab.ui.control.DropDown
        EdgeThreshSlider    matlab.ui.control.Slider
        EdgeThreshValueLbl  matlab.ui.control.Label
        EdgeSigmaSpinner    matlab.ui.control.Spinner
        EdgeRadiusSpinner   matlab.ui.control.Spinner
        EdgeMinAreaSpinner  matlab.ui.control.Spinner
        EdgeRunBtn          matlab.ui.control.Button
        EdgeViewPanel       matlab.ui.container.Panel
        EdgeViewGrid        matlab.ui.container.GridLayout
        EdgeAxOrig          matlab.ui.control.UIAxes
        EdgeAxEdge          matlab.ui.control.UIAxes
        EdgeAxMask          matlab.ui.control.UIAxes
        EdgeAxSeg           matlab.ui.control.UIAxes
        EdgeStatusLabel     matlab.ui.control.Label

        % --- Tab 2: Split and Merge ---
        SplitTab            matlab.ui.container.Tab
        SplitTabGrid        matlab.ui.container.GridLayout
        SplitCtrlPanel      matlab.ui.container.Panel
        SplitCtrlGrid       matlab.ui.container.GridLayout
        SplitLoadBtn        matlab.ui.control.Button
        SplitUseTab1Btn     matlab.ui.control.Button
        SplitFileLabel      matlab.ui.control.Label
        SplitThreshSlider   matlab.ui.control.Slider
        SplitThreshValLbl   matlab.ui.control.Label
        MergeThreshSlider   matlab.ui.control.Slider
        MergeThreshValLbl   matlab.ui.control.Label
        MinBlockDropDown    matlab.ui.control.DropDown
        SplitRunBtn         matlab.ui.control.Button
        SplitViewPanel      matlab.ui.container.Panel
        SplitViewGrid       matlab.ui.container.GridLayout
        SplitAxOrig         matlab.ui.control.UIAxes
        SplitAxSplit        matlab.ui.control.UIAxes
        SplitAxMerged       matlab.ui.control.UIAxes
        SplitStatusLabel    matlab.ui.control.Label

        % --- Tab 3: Semantic Segmentation (Deep Learning) ---
        DLTab               matlab.ui.container.Tab
        DLTabGrid           matlab.ui.container.GridLayout
        DLCtrlPanel         matlab.ui.container.Panel
        DLCtrlGrid          matlab.ui.container.GridLayout
        DLLoadBtn           matlab.ui.control.Button
        DLFileLabel         matlab.ui.control.Label
        DLRunBtn            matlab.ui.control.Button
        DLOpacitySlider     matlab.ui.control.Slider
        DLOpacityValLbl     matlab.ui.control.Label
        DLClassSummaryArea  matlab.ui.control.TextArea
        DLViewPanel         matlab.ui.container.Panel
        DLViewGrid          matlab.ui.container.GridLayout
        DLAxOrig            matlab.ui.control.UIAxes
        DLAxColorMask       matlab.ui.control.UIAxes
        DLAxOverlay         matlab.ui.control.UIAxes
        DLAxLegend          matlab.ui.control.UIAxes
        DLStatusLabel       matlab.ui.control.Label
    end

    properties (Access = private)
        ProjectPaths        struct
        CurrentImageEdge    uint8
        CurrentPathEdge     char = ''
        CurrentImageSplit   uint8
        CurrentPathSplit    char = ''
        CurrentImageDL      uint8
        CurrentPathDL       char = ''
        LastColorMask       uint8
        LastClassPercent    struct

        % Palet visual terpadu (Precision Lab: Netral + Aksen Cobalt tunggal)
        ColorCanvas         double = [0.965, 0.970, 0.978]
        ColorSurface        double = [1.000, 1.000, 1.000]
        ColorBorder         double = [0.865, 0.885, 0.915]
        ColorInk            double = [0.094, 0.110, 0.137]
        ColorMuted          double = [0.360, 0.410, 0.480]
        ColorAccent         double = [0.114, 0.306, 0.847]
        ColorSecondaryBtn   double = [0.940, 0.950, 0.968]
    end

    methods (Access = private)

        function startupFcn(app)
            % Resolusi path relatif otomatis berbasis lokasi file ini
            app.ProjectPaths = gui_helpers.resolve_project_paths();

            % Inisialisasi placeholder awal di seluruh axes dan legend Tab 3
            app.setEmptyAxes(app.EdgeAxOrig, 'Citra Asli');
            app.setEmptyAxes(app.EdgeAxEdge, 'Hasil Deteksi Tepi');
            app.setEmptyAxes(app.EdgeAxMask, 'Mask Biner Objek');
            app.setEmptyAxes(app.EdgeAxSeg, 'Objek Terisolasi');

            app.setEmptyAxes(app.SplitAxOrig, 'Citra Asli');
            app.setEmptyAxes(app.SplitAxSplit, 'Hasil Tahap Split');
            app.setEmptyAxes(app.SplitAxMerged, 'Hasil Tahap Merge');

            app.setEmptyAxes(app.DLAxOrig, 'Citra Masukan');
            app.setEmptyAxes(app.DLAxColorMask, 'Peta Segmentasi Semantik');
            app.setEmptyAxes(app.DLAxOverlay, 'Overlay Segmentasi');
            gui_helpers.render_legend(app.DLAxLegend);
        end

        function setEmptyAxes(app, ax, titleStr)
            cla(ax, 'reset');
            ax.XLim = [0, 1];
            ax.YLim = [0, 1];
            axis(ax, 'off');
            ax.Color = [0.980, 0.984, 0.990];
            text(ax, 0.5, 0.5, 'Muat citra untuk memulai', ...
                'Units', 'normalized', ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', 9.5, ...
                'Color', app.ColorMuted);
            title(ax, titleStr, 'FontSize', 10.5, 'FontWeight', 'bold', 'Color', app.ColorInk);
        end

        function showImageOnAxes(app, ax, img, titleStr)
            cla(ax, 'reset');
            imshow(img, 'Parent', ax);
            title(ax, titleStr, 'FontSize', 10.5, 'FontWeight', 'bold', 'Color', app.ColorInk);
        end

        function styleSurfacePanel(app, panel)
            panel.BackgroundColor = app.ColorSurface;
            panel.BorderType = 'line';
            % BorderColor tersedia sejak MATLAB R2023a; aman diabaikan pada versi lebih lama
            if isprop(panel, 'BorderColor')
                panel.BorderColor = app.ColorBorder;
            end
        end

        function val = readSliderValue(~, src, event)
            if nargin >= 3 && ~isempty(event) && isprop(event, 'Value')
                val = event.Value;
            else
                val = src.Value;
            end
        end

        function [img, fullPath, fileName] = promptLoadImage(app, defaultDir)
            img = [];
            fullPath = '';
            fileName = '';

            startPath = app.ProjectPaths.projectRoot;
            if isfolder(defaultDir)
                startPath = defaultDir;
            end

            filterSpec = {'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', 'File Citra (*.jpg, *.png, *.bmp, *.tif)'};
            [file, folder] = uigetfile(filterSpec, 'Pilih Citra Masukan', startPath);
            figure(app.UIFigure);

            if isequal(file, 0)
                return;
            end

            fullPath = fullfile(folder, file);
            fileName = file;
            rawImg = imread(fullPath);
            img = gui_helpers.normalize_image_uint8(rawImg);
        end

        % =================================================================
        % CALLBACK TAB 1: DETEKSI TEPI & SEGMENTASI OBJEK
        % =================================================================

        function onEdgeLoad(app, ~, ~)
            [img, fullPath, fileName] = app.promptLoadImage(app.ProjectPaths.bsdsDir);
            if isempty(img)
                return;
            end

            app.CurrentImageEdge = img;
            app.CurrentPathEdge = fullPath;
            app.EdgeFileLabel.Text = sprintf('File: %s (%dx%d)', fileName, size(img, 2), size(img, 1));
            app.showImageOnAxes(app.EdgeAxOrig, img, 'Citra Asli');
            app.setEmptyAxes(app.EdgeAxEdge, 'Hasil Deteksi Tepi');
            app.setEmptyAxes(app.EdgeAxMask, 'Mask Biner Objek');
            app.setEmptyAxes(app.EdgeAxSeg, 'Objek Terisolasi');
            app.EdgeStatusLabel.Text = 'Status: Citra dimuat. Atur parameter lalu klik Proses.';
        end

        function onEdgeThreshChanged(app, src, event)
            val = app.readSliderValue(src, event);
            app.EdgeThreshValueLbl.Text = sprintf('%.2f', val);
        end

        function onEdgeRun(app, ~, ~)
            if isempty(app.CurrentImageEdge)
                uialert(app.UIFigure, 'Silakan muat citra terlebih dahulu pada Tab 1.', 'Citra Belum Dipilih');
                return;
            end

            dlg = uiprogressdlg(app.UIFigure, ...
                'Title', 'Deteksi Tepi', ...
                'Message', 'Menjalankan deteksi tepi dan segmentasi objek...', ...
                'Indeterminate', 'on');
            try
                params = struct();
                params.thresh = app.EdgeThreshSlider.Value;
                params.sigma = app.EdgeSigmaSpinner.Value;
                params.morphRadius = round(app.EdgeRadiusSpinner.Value);
                params.minArea = round(app.EdgeMinAreaSpinner.Value);

                operator = app.EdgeOpDropDown.Value;
                tStart = tic;
                [edgeMap, objMask, segObjImg] = run_edge_segmentation( ...
                    app.CurrentImageEdge, operator, params);
                elapsed = toc(tStart);

                app.showImageOnAxes(app.EdgeAxOrig, app.CurrentImageEdge, 'Citra Asli');
                app.showImageOnAxes(app.EdgeAxEdge, edgeMap, sprintf('Deteksi Tepi (%s)', operator));
                app.showImageOnAxes(app.EdgeAxMask, objMask, 'Mask Biner Objek');
                app.showImageOnAxes(app.EdgeAxSeg, segObjImg, 'Objek Terisolasi');

                app.EdgeStatusLabel.Text = sprintf( ...
                    'Status: Selesai dalam %.3f dtk (Operator: %s)', elapsed, operator);
            catch ME
                if isvalid(dlg), close(dlg); end
                uialert(app.UIFigure, ME.message, 'Gagal Memproses Tab 1');
                app.EdgeStatusLabel.Text = ['Status: Error - ' ME.message];
                return;
            end
            if isvalid(dlg), close(dlg); end
        end

        % =================================================================
        % CALLBACK TAB 2: SPLIT AND MERGE
        % =================================================================

        function onSplitLoad(app, ~, ~)
            [img, fullPath, fileName] = app.promptLoadImage(app.ProjectPaths.bsdsDir);
            if isempty(img)
                return;
            end

            app.CurrentImageSplit = img;
            app.CurrentPathSplit = fullPath;
            app.SplitFileLabel.Text = sprintf('File: %s (%dx%d)', fileName, size(img, 2), size(img, 1));
            app.showImageOnAxes(app.SplitAxOrig, img, 'Citra Asli');
            app.setEmptyAxes(app.SplitAxSplit, 'Hasil Tahap Split');
            app.setEmptyAxes(app.SplitAxMerged, 'Hasil Tahap Merge');
            app.SplitStatusLabel.Text = 'Status: Citra dimuat. Sesuaikan threshold lalu jalankan Split & Merge.';
        end

        function onSplitUseTab1(app, ~, ~)
            if isempty(app.CurrentImageEdge)
                uialert(app.UIFigure, 'Belum ada citra yang dimuat di Tab 1.', 'Informasi');
                return;
            end

            app.CurrentImageSplit = app.CurrentImageEdge;
            app.CurrentPathSplit = app.CurrentPathEdge;
            [~, name, ext] = fileparts(app.CurrentPathSplit);
            if isempty(name)
                fileStr = 'Citra Tab 1';
            else
                fileStr = [name ext];
            end
            app.SplitFileLabel.Text = sprintf('File: %s (%dx%d)', ...
                fileStr, size(app.CurrentImageSplit, 2), size(app.CurrentImageSplit, 1));
            app.showImageOnAxes(app.SplitAxOrig, app.CurrentImageSplit, 'Citra Asli');
            app.setEmptyAxes(app.SplitAxSplit, 'Hasil Tahap Split');
            app.setEmptyAxes(app.SplitAxMerged, 'Hasil Tahap Merge');
            app.SplitStatusLabel.Text = 'Status: Menggunakan citra aktif dari Tab 1.';
        end

        function onSplitThreshChanged(app, src, event)
            val = app.readSliderValue(src, event);
            app.SplitThreshValLbl.Text = sprintf('%.0f', val);
        end

        function onMergeThreshChanged(app, src, event)
            val = app.readSliderValue(src, event);
            app.MergeThreshValLbl.Text = sprintf('%.0f', val);
        end

        function onSplitRun(app, ~, ~)
            if isempty(app.CurrentImageSplit)
                uialert(app.UIFigure, 'Silakan muat citra terlebih dahulu pada Tab 2.', 'Citra Belum Dipilih');
                return;
            end

            dlg = uiprogressdlg(app.UIFigure, ...
                'Title', 'Split and Merge', ...
                'Message', 'Menjalankan segmentasi region Split and Merge...', ...
                'Indeterminate', 'on');
            try
                splitThresh = app.SplitThreshSlider.Value;
                mergeThresh = app.MergeThreshSlider.Value;
                minBlock = str2double(app.MinBlockDropDown.Value);

                tStart = tic;
                [splitVisImg, mergedImg, stats] = split_and_merge( ...
                    app.CurrentImageSplit, splitThresh, mergeThresh, minBlock);
                elapsed = toc(tStart);

                app.showImageOnAxes(app.SplitAxOrig, app.CurrentImageSplit, 'Citra Asli');
                app.showImageOnAxes(app.SplitAxSplit, splitVisImg, 'After Split (Quadtree)');
                app.showImageOnAxes(app.SplitAxMerged, mergedImg, 'After Merge (Hasil Akhir)');

                regionMsg = '';
                if isstruct(stats) && isfield(stats, 'numSplitRegions') && isfield(stats, 'numMergedRegions')
                    regionMsg = sprintf(' | Region: %d -> %d', stats.numSplitRegions, stats.numMergedRegions);
                end

                app.SplitStatusLabel.Text = sprintf('Status: Selesai dalam %.3f dtk%s', elapsed, regionMsg);
            catch ME
                if isvalid(dlg), close(dlg); end
                uialert(app.UIFigure, ME.message, 'Gagal Memproses Tab 2');
                app.SplitStatusLabel.Text = ['Status: Error - ' ME.message];
                return;
            end
            if isvalid(dlg), close(dlg); end
        end

        % =================================================================
        % CALLBACK TAB 3: SEMANTIC SEGMENTATION (DEEP LEARNING)
        % =================================================================

        function onDLLoad(app, ~, ~)
            [img, fullPath, fileName] = app.promptLoadImage(app.ProjectPaths.camvidTestDir);
            if isempty(img)
                return;
            end

            app.CurrentImageDL = img;
            app.CurrentPathDL = fullPath;
            app.LastColorMask = uint8([]);
            app.LastClassPercent = struct();

            app.DLFileLabel.Text = sprintf('File: %s (%dx%d)', fileName, size(img, 2), size(img, 1));
            app.showImageOnAxes(app.DLAxOrig, img, 'Citra Masukan');
            app.setEmptyAxes(app.DLAxColorMask, 'Peta Segmentasi Semantik');
            app.setEmptyAxes(app.DLAxOverlay, 'Overlay Segmentasi');
            gui_helpers.render_legend(app.DLAxLegend);
            app.DLClassSummaryArea.Value = {'Belum ada hasil inferensi.'};
            app.DLStatusLabel.Text = 'Status: Citra CamVid dimuat. Klik Jalankan Inferensi untuk memproses.';
        end

        function onDLRun(app, ~, ~)
            if isempty(app.CurrentImageDL)
                uialert(app.UIFigure, 'Silakan muat citra terlebih dahulu pada Tab 3.', 'Citra Belum Dipilih');
                return;
            end

            dlg = uiprogressdlg(app.UIFigure, ...
                'Title', 'Semantic Segmentation', ...
                'Message', 'Memuat model lokal DeepLabv3+ dan menjalankan inferensi...', ...
                'Indeterminate', 'on');
            try
                opacity = app.DLOpacitySlider.Value / 100.0;
                tStart = tic;
                res = gui_helpers.run_dl_segmentation( ...
                    app.CurrentPathDL, app.DLFileLabel.Text, opacity, app.ProjectPaths);
                elapsed = toc(tStart);

                % Cache colorMask agar slider opacity bekerja cepat tanpa re-inferensi
                app.LastColorMask = res.colorMask;
                app.LastClassPercent = res.classPercentages;

                blended = gui_helpers.blend_overlay(app.CurrentImageDL, app.LastColorMask, opacity);
                app.showImageOnAxes(app.DLAxOrig, app.CurrentImageDL, 'Citra Masukan');
                app.showImageOnAxes(app.DLAxColorMask, app.LastColorMask, 'Color Mask (11 Kelas)');
                app.showImageOnAxes(app.DLAxOverlay, blended, sprintf('Overlay (Opacity %d%%)', round(opacity * 100)));
                gui_helpers.render_legend(app.DLAxLegend, app.LastClassPercent);

                app.updateClassSummary(app.LastClassPercent);
                app.DLStatusLabel.Text = sprintf( ...
                    'Status: Inferensi offline selesai dalam %.2f detik (DeepLabv3+ MobileNetV3-Large).', elapsed);
            catch ME
                if isvalid(dlg), close(dlg); end
                uialert(app.UIFigure, ME.message, 'Gagal Inferensi Deep Learning');
                app.DLStatusLabel.Text = ['Status: Error - ' ME.message];
                return;
            end
            if isvalid(dlg), close(dlg); end
        end

        function onDLOpacityChanged(app, src, event)
            pct = round(app.readSliderValue(src, event));
            app.DLOpacityValLbl.Text = sprintf('%d%%', pct);

            % Update overlay secara lokal jika colorMask sudah tersedia di cache
            if ~isempty(app.CurrentImageDL) && ~isempty(app.LastColorMask)
                blended = gui_helpers.blend_overlay(app.CurrentImageDL, app.LastColorMask, pct / 100.0);
                app.showImageOnAxes(app.DLAxOverlay, blended, sprintf('Overlay (Opacity %d%%)', pct));
            end
        end

        function updateClassSummary(app, classPct)
            if ~isstruct(classPct) || isempty(fieldnames(classPct))
                app.DLClassSummaryArea.Value = {'Ringkasan piksel tidak tersedia.'};
                return;
            end

            classes = gui_helpers.get_camvid_classes();
            lines = cell(1, numel(classes));
            for i = 1:numel(classes)
                cname = classes(i).name;
                val = 0;
                if isfield(classPct, cname)
                    val = classPct.(cname);
                end
                lines{i} = sprintf('%-12s : %5.2f%%', cname, val);
            end
            app.DLClassSummaryArea.Value = lines;
        end

    end

    methods (Access = private)

        function createComponents(app)
            % Jendela utama dengan proporsi lapang dan latar belakang netral laboratorium
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [80, 60, 1280, 780];
            app.UIFigure.Name = 'IF4073 Pemrosesan Citra Digital - Segmentasi Citra Terpadu';
            app.UIFigure.Color = app.ColorCanvas;

            app.MainGrid = uigridlayout(app.UIFigure, [2, 1]);
            app.MainGrid.RowHeight = {56, '1x'};
            app.MainGrid.Padding = [14, 14, 14, 14];
            app.MainGrid.RowSpacing = 10;
            app.MainGrid.BackgroundColor = app.ColorCanvas;

            % Header ringkas satu baris judul + subjudul
            app.HeaderPanel = uipanel(app.MainGrid);
            app.styleSurfacePanel(app.HeaderPanel);
            headerGrid = uigridlayout(app.HeaderPanel, [1, 2]);
            headerGrid.ColumnWidth = {'1x', 'fit'};
            headerGrid.Padding = [16, 10, 16, 10];
            headerGrid.BackgroundColor = app.ColorSurface;

            app.AppTitleLabel = uilabel(headerGrid);
            app.AppTitleLabel.Text = 'Workstation Segmentasi Citra — IF4073';
            app.AppTitleLabel.FontName = 'Segoe UI';
            app.AppTitleLabel.FontSize = 15;
            app.AppTitleLabel.FontWeight = 'bold';
            app.AppTitleLabel.FontColor = app.ColorInk;

            app.AppSubtitleLabel = uilabel(headerGrid);
            app.AppSubtitleLabel.Text = 'Deteksi Tepi  |  Split & Merge  |  DeepLabv3+ CamVid (11 Kelas)';
            app.AppSubtitleLabel.FontName = 'Segoe UI';
            app.AppSubtitleLabel.FontSize = 10;
            app.AppSubtitleLabel.FontColor = app.ColorMuted;
            app.AppSubtitleLabel.HorizontalAlignment = 'right';

            % TabGroup utama
            app.TabGroup = uitabgroup(app.MainGrid);

            app.buildEdgeTab();
            app.buildSplitTab();
            app.buildDLTab();

            app.UIFigure.Visible = 'on';
        end

        function buildEdgeTab(app)
            app.EdgeTab = uitab(app.TabGroup, 'Title', '1. Deteksi Tepi & Objek');
            app.EdgeTab.BackgroundColor = app.ColorCanvas;

            app.EdgeTabGrid = uigridlayout(app.EdgeTab, [1, 2]);
            app.EdgeTabGrid.ColumnWidth = {290, '1x'};
            app.EdgeTabGrid.Padding = [12, 12, 12, 12];
            app.EdgeTabGrid.ColumnSpacing = 12;
            app.EdgeTabGrid.BackgroundColor = app.ColorCanvas;

            % Panel Kontrol Kiri
            app.EdgeCtrlPanel = uipanel(app.EdgeTabGrid);
            app.styleSurfacePanel(app.EdgeCtrlPanel);

            app.EdgeCtrlGrid = uigridlayout(app.EdgeCtrlPanel, [13, 2]);
            app.EdgeCtrlGrid.ColumnWidth = {'1x', 62};
            app.EdgeCtrlGrid.RowHeight = {22, 36, 22, 20, 32, 20, 30, 20, 30, 20, 30, '1x', 40};
            app.EdgeCtrlGrid.Padding = [14, 14, 14, 14];
            app.EdgeCtrlGrid.RowSpacing = 8;
            app.EdgeCtrlGrid.BackgroundColor = app.ColorSurface;

            lbl = uilabel(app.EdgeCtrlGrid, 'Text', 'DATA MASUKAN', 'FontSize', 10, 'FontWeight', 'bold', 'FontColor', app.ColorMuted);
            lbl.Layout.Row = 1; lbl.Layout.Column = [1, 2];

            app.EdgeLoadBtn = uibutton(app.EdgeCtrlGrid, 'push', 'Text', 'Pilih Citra', ...
                'ButtonPushedFcn', @app.onEdgeLoad, ...
                'BackgroundColor', app.ColorSecondaryBtn, 'FontColor', app.ColorInk, ...
                'FontWeight', 'bold', 'FontSize', 10.5);
            app.EdgeLoadBtn.Layout.Row = 2; app.EdgeLoadBtn.Layout.Column = [1, 2];

            app.EdgeFileLabel = uilabel(app.EdgeCtrlGrid, 'Text', 'File: belum dipilih', ...
                'FontSize', 9.5, 'FontColor', app.ColorMuted);
            app.EdgeFileLabel.Layout.Row = 3; app.EdgeFileLabel.Layout.Column = [1, 2];

            lblOp = uilabel(app.EdgeCtrlGrid, 'Text', 'Operator Deteksi Tepi', ...
                'FontSize', 10, 'FontWeight', 'bold', 'FontColor', app.ColorInk);
            lblOp.Layout.Row = 4; lblOp.Layout.Column = [1, 2];

            app.EdgeOpDropDown = uidropdown(app.EdgeCtrlGrid, ...
                'Items', {'Sobel', 'Prewitt', 'Roberts', 'Laplace', 'LoG', 'Canny'}, ...
                'Value', 'Sobel', 'FontSize', 10.5);
            app.EdgeOpDropDown.Layout.Row = 5; app.EdgeOpDropDown.Layout.Column = [1, 2];

            lblTh = uilabel(app.EdgeCtrlGrid, 'Text', 'Threshold Tepi', ...
                'FontSize', 10, 'FontWeight', 'bold', 'FontColor', app.ColorInk);
            lblTh.Layout.Row = 6; lblTh.Layout.Column = 1;

            app.EdgeThreshValueLbl = uilabel(app.EdgeCtrlGrid, 'Text', '0.25', ...
                'HorizontalAlignment', 'right', 'FontSize', 10, 'FontColor', app.ColorAccent, 'FontWeight', 'bold');
            app.EdgeThreshValueLbl.Layout.Row = 6; app.EdgeThreshValueLbl.Layout.Column = 2;

            app.EdgeThreshSlider = uislider(app.EdgeCtrlGrid, 'Limits', [0.02, 0.95], 'Value', 0.25, ...
                'ValueChangingFcn', @app.onEdgeThreshChanged, ...
                'ValueChangedFcn', @app.onEdgeThreshChanged);
            app.EdgeThreshSlider.Layout.Row = 7; app.EdgeThreshSlider.Layout.Column = [1, 2];

            lblSig = uilabel(app.EdgeCtrlGrid, 'Text', 'Sigma (LoG / Canny)', ...
                'FontSize', 10, 'FontColor', app.ColorInk);
            lblSig.Layout.Row = 8; lblSig.Layout.Column = 1;
            app.EdgeSigmaSpinner = uispinner(app.EdgeCtrlGrid, 'Limits', [0.3, 5.0], 'Step', 0.1, 'Value', 1.0);
            app.EdgeSigmaSpinner.Layout.Row = 9; app.EdgeSigmaSpinner.Layout.Column = [1, 2];

            lblRad = uilabel(app.EdgeCtrlGrid, 'Text', 'Radius Closing & Min Area', ...
                'FontSize', 10, 'FontColor', app.ColorInk);
            lblRad.Layout.Row = 10; lblRad.Layout.Column = [1, 2];

            app.EdgeRadiusSpinner = uispinner(app.EdgeCtrlGrid, 'Limits', [1, 25], 'Step', 1, 'Value', 3);
            app.EdgeRadiusSpinner.Layout.Row = 11; app.EdgeRadiusSpinner.Layout.Column = 2;

            app.EdgeMinAreaSpinner = uispinner(app.EdgeCtrlGrid, 'Limits', [10, 50000], 'Step', 50, 'Value', 500);
            app.EdgeMinAreaSpinner.Layout.Row = 11; app.EdgeMinAreaSpinner.Layout.Column = 1;

            app.EdgeRunBtn = uibutton(app.EdgeCtrlGrid, 'push', 'Text', 'Proses Segmentasi', ...
                'ButtonPushedFcn', @app.onEdgeRun, ...
                'BackgroundColor', app.ColorAccent, 'FontColor', [1, 1, 1], ...
                'FontWeight', 'bold', 'FontSize', 11);
            app.EdgeRunBtn.Layout.Row = 13; app.EdgeRunBtn.Layout.Column = [1, 2];

            % Panel Visualisasi Kanan (Grid 2x2 + Status)
            app.EdgeViewPanel = uipanel(app.EdgeTabGrid);
            app.styleSurfacePanel(app.EdgeViewPanel);

            app.EdgeViewGrid = uigridlayout(app.EdgeViewPanel, [3, 2]);
            app.EdgeViewGrid.RowHeight = {'1x', '1x', 24};
            app.EdgeViewGrid.Padding = [12, 12, 12, 10];
            app.EdgeViewGrid.RowSpacing = 10;
            app.EdgeViewGrid.ColumnSpacing = 12;
            app.EdgeViewGrid.BackgroundColor = app.ColorSurface;

            app.EdgeAxOrig = uiaxes(app.EdgeViewGrid);
            app.EdgeAxOrig.Layout.Row = 1; app.EdgeAxOrig.Layout.Column = 1;

            app.EdgeAxEdge = uiaxes(app.EdgeViewGrid);
            app.EdgeAxEdge.Layout.Row = 1; app.EdgeAxEdge.Layout.Column = 2;

            app.EdgeAxMask = uiaxes(app.EdgeViewGrid);
            app.EdgeAxMask.Layout.Row = 2; app.EdgeAxMask.Layout.Column = 1;

            app.EdgeAxSeg = uiaxes(app.EdgeViewGrid);
            app.EdgeAxSeg.Layout.Row = 2; app.EdgeAxSeg.Layout.Column = 2;

            app.EdgeStatusLabel = uilabel(app.EdgeViewGrid, 'Text', 'Status: Siap.', ...
                'FontSize', 9.5, 'FontColor', app.ColorMuted);
            app.EdgeStatusLabel.Layout.Row = 3; app.EdgeStatusLabel.Layout.Column = [1, 2];
        end

        function buildSplitTab(app)
            app.SplitTab = uitab(app.TabGroup, 'Title', '2. Split and Merge');
            app.SplitTab.BackgroundColor = app.ColorCanvas;

            app.SplitTabGrid = uigridlayout(app.SplitTab, [1, 2]);
            app.SplitTabGrid.ColumnWidth = {290, '1x'};
            app.SplitTabGrid.Padding = [12, 12, 12, 12];
            app.SplitTabGrid.ColumnSpacing = 12;
            app.SplitTabGrid.BackgroundColor = app.ColorCanvas;

            % Panel Kontrol Kiri
            app.SplitCtrlPanel = uipanel(app.SplitTabGrid);
            app.styleSurfacePanel(app.SplitCtrlPanel);

            app.SplitCtrlGrid = uigridlayout(app.SplitCtrlPanel, [12, 2]);
            app.SplitCtrlGrid.ColumnWidth = {'1x', 60};
            app.SplitCtrlGrid.RowHeight = {22, 36, 34, 22, 20, 34, 20, 34, 20, 32, '1x', 40};
            app.SplitCtrlGrid.Padding = [14, 14, 14, 14];
            app.SplitCtrlGrid.RowSpacing = 8;
            app.SplitCtrlGrid.BackgroundColor = app.ColorSurface;

            lbl = uilabel(app.SplitCtrlGrid, 'Text', 'DATA MASUKAN', 'FontSize', 10, 'FontWeight', 'bold', 'FontColor', app.ColorMuted);
            lbl.Layout.Row = 1; lbl.Layout.Column = [1, 2];

            app.SplitLoadBtn = uibutton(app.SplitCtrlGrid, 'push', 'Text', 'Pilih Citra', ...
                'ButtonPushedFcn', @app.onSplitLoad, ...
                'BackgroundColor', app.ColorSecondaryBtn, 'FontColor', app.ColorInk, ...
                'FontWeight', 'bold', 'FontSize', 10.5);
            app.SplitLoadBtn.Layout.Row = 2; app.SplitLoadBtn.Layout.Column = [1, 2];

            app.SplitUseTab1Btn = uibutton(app.SplitCtrlGrid, 'push', 'Text', 'Pakai Citra Tab 1', ...
                'ButtonPushedFcn', @app.onSplitUseTab1, ...
                'BackgroundColor', app.ColorSurface, 'FontColor', app.ColorAccent, ...
                'FontWeight', 'bold', 'FontSize', 10);
            app.SplitUseTab1Btn.Layout.Row = 3; app.SplitUseTab1Btn.Layout.Column = [1, 2];

            app.SplitFileLabel = uilabel(app.SplitCtrlGrid, 'Text', 'File: belum dipilih', ...
                'FontSize', 9.5, 'FontColor', app.ColorMuted);
            app.SplitFileLabel.Layout.Row = 4; app.SplitFileLabel.Layout.Column = [1, 2];

            lblST = uilabel(app.SplitCtrlGrid, 'Text', 'Threshold Split', ...
                'FontSize', 10, 'FontWeight', 'bold', 'FontColor', app.ColorInk);
            lblST.Layout.Row = 5; lblST.Layout.Column = 1;

            app.SplitThreshValLbl = uilabel(app.SplitCtrlGrid, 'Text', '25', ...
                'HorizontalAlignment', 'right', 'FontSize', 10, 'FontColor', app.ColorAccent, 'FontWeight', 'bold');
            app.SplitThreshValLbl.Layout.Row = 5; app.SplitThreshValLbl.Layout.Column = 2;

            app.SplitThreshSlider = uislider(app.SplitCtrlGrid, 'Limits', [5, 100], 'Value', 25, ...
                'ValueChangingFcn', @app.onSplitThreshChanged, ...
                'ValueChangedFcn', @app.onSplitThreshChanged);
            app.SplitThreshSlider.Layout.Row = 6; app.SplitThreshSlider.Layout.Column = [1, 2];

            lblMT = uilabel(app.SplitCtrlGrid, 'Text', 'Threshold Merge', ...
                'FontSize', 10, 'FontWeight', 'bold', 'FontColor', app.ColorInk);
            lblMT.Layout.Row = 7; lblMT.Layout.Column = 1;

            app.MergeThreshValLbl = uilabel(app.SplitCtrlGrid, 'Text', '15', ...
                'HorizontalAlignment', 'right', 'FontSize', 10, 'FontColor', app.ColorAccent, 'FontWeight', 'bold');
            app.MergeThreshValLbl.Layout.Row = 7; app.MergeThreshValLbl.Layout.Column = 2;

            app.MergeThreshSlider = uislider(app.SplitCtrlGrid, 'Limits', [5, 100], 'Value', 15, ...
                'ValueChangingFcn', @app.onMergeThreshChanged, ...
                'ValueChangedFcn', @app.onMergeThreshChanged);
            app.MergeThreshSlider.Layout.Row = 8; app.MergeThreshSlider.Layout.Column = [1, 2];

            lblBlk = uilabel(app.SplitCtrlGrid, 'Text', 'Ukuran Blok Minimum (px)', ...
                'FontSize', 10, 'FontWeight', 'bold', 'FontColor', app.ColorInk);
            lblBlk.Layout.Row = 9; lblBlk.Layout.Column = [1, 2];

            app.MinBlockDropDown = uidropdown(app.SplitCtrlGrid, ...
                'Items', {'2', '4', '8', '16'}, 'Value', '8', 'FontSize', 10.5);
            app.MinBlockDropDown.Layout.Row = 10; app.MinBlockDropDown.Layout.Column = [1, 2];

            app.SplitRunBtn = uibutton(app.SplitCtrlGrid, 'push', 'Text', 'Jalankan Split & Merge', ...
                'ButtonPushedFcn', @app.onSplitRun, ...
                'BackgroundColor', app.ColorAccent, 'FontColor', [1, 1, 1], ...
                'FontWeight', 'bold', 'FontSize', 11);
            app.SplitRunBtn.Layout.Row = 12; app.SplitRunBtn.Layout.Column = [1, 2];

            % Panel Visualisasi Kanan (3 Panel Citra + Status)
            app.SplitViewPanel = uipanel(app.SplitTabGrid);
            app.styleSurfacePanel(app.SplitViewPanel);

            app.SplitViewGrid = uigridlayout(app.SplitViewPanel, [2, 3]);
            app.SplitViewGrid.RowHeight = {'1x', 24};
            app.SplitViewGrid.Padding = [12, 12, 12, 10];
            app.SplitViewGrid.RowSpacing = 10;
            app.SplitViewGrid.ColumnSpacing = 10;
            app.SplitViewGrid.BackgroundColor = app.ColorSurface;

            app.SplitAxOrig = uiaxes(app.SplitViewGrid);
            app.SplitAxOrig.Layout.Row = 1; app.SplitAxOrig.Layout.Column = 1;

            app.SplitAxSplit = uiaxes(app.SplitViewGrid);
            app.SplitAxSplit.Layout.Row = 1; app.SplitAxSplit.Layout.Column = 2;

            app.SplitAxMerged = uiaxes(app.SplitViewGrid);
            app.SplitAxMerged.Layout.Row = 1; app.SplitAxMerged.Layout.Column = 3;

            app.SplitStatusLabel = uilabel(app.SplitViewGrid, 'Text', 'Status: Siap.', ...
                'FontSize', 9.5, 'FontColor', app.ColorMuted);
            app.SplitStatusLabel.Layout.Row = 2; app.SplitStatusLabel.Layout.Column = [1, 3];
        end

        function buildDLTab(app)
            app.DLTab = uitab(app.TabGroup, 'Title', '3. Semantic Segmentation (CNN)');
            app.DLTab.BackgroundColor = app.ColorCanvas;

            app.DLTabGrid = uigridlayout(app.DLTab, [1, 2]);
            app.DLTabGrid.ColumnWidth = {290, '1x'};
            app.DLTabGrid.Padding = [12, 12, 12, 12];
            app.DLTabGrid.ColumnSpacing = 12;
            app.DLTabGrid.BackgroundColor = app.ColorCanvas;

            % Panel Kontrol Kiri
            app.DLCtrlPanel = uipanel(app.DLTabGrid);
            app.styleSurfacePanel(app.DLCtrlPanel);

            app.DLCtrlGrid = uigridlayout(app.DLCtrlPanel, [9, 2]);
            app.DLCtrlGrid.ColumnWidth = {'1x', 56};
            app.DLCtrlGrid.RowHeight = {22, 36, 22, 40, 20, 34, 20, '1x', 18};
            app.DLCtrlGrid.Padding = [14, 14, 14, 14];
            app.DLCtrlGrid.RowSpacing = 8;
            app.DLCtrlGrid.BackgroundColor = app.ColorSurface;

            lbl = uilabel(app.DLCtrlGrid, 'Text', 'DATA UJI CAMVID', 'FontSize', 10, 'FontWeight', 'bold', 'FontColor', app.ColorMuted);
            lbl.Layout.Row = 1; lbl.Layout.Column = [1, 2];

            app.DLLoadBtn = uibutton(app.DLCtrlGrid, 'push', 'Text', 'Pilih Citra', ...
                'ButtonPushedFcn', @app.onDLLoad, ...
                'BackgroundColor', app.ColorSecondaryBtn, 'FontColor', app.ColorInk, ...
                'FontWeight', 'bold', 'FontSize', 10.5);
            app.DLLoadBtn.Layout.Row = 2; app.DLLoadBtn.Layout.Column = [1, 2];

            app.DLFileLabel = uilabel(app.DLCtrlGrid, 'Text', 'File: belum dipilih', ...
                'FontSize', 9.5, 'FontColor', app.ColorMuted);
            app.DLFileLabel.Layout.Row = 3; app.DLFileLabel.Layout.Column = [1, 2];

            app.DLRunBtn = uibutton(app.DLCtrlGrid, 'push', 'Text', 'Jalankan Inferensi', ...
                'ButtonPushedFcn', @app.onDLRun, ...
                'BackgroundColor', app.ColorAccent, 'FontColor', [1, 1, 1], ...
                'FontWeight', 'bold', 'FontSize', 11);
            app.DLRunBtn.Layout.Row = 4; app.DLRunBtn.Layout.Column = [1, 2];

            lblOp = uilabel(app.DLCtrlGrid, 'Text', 'Opacity Overlay', ...
                'FontSize', 10, 'FontWeight', 'bold', 'FontColor', app.ColorInk);
            lblOp.Layout.Row = 5; lblOp.Layout.Column = 1;

            app.DLOpacityValLbl = uilabel(app.DLCtrlGrid, 'Text', '50%', ...
                'HorizontalAlignment', 'right', 'FontSize', 10, 'FontColor', app.ColorAccent, 'FontWeight', 'bold');
            app.DLOpacityValLbl.Layout.Row = 5; app.DLOpacityValLbl.Layout.Column = 2;

            app.DLOpacitySlider = uislider(app.DLCtrlGrid, 'Limits', [0, 100], 'Value', 50, ...
                'ValueChangingFcn', @app.onDLOpacityChanged, ...
                'ValueChangedFcn', @app.onDLOpacityChanged);
            app.DLOpacitySlider.Layout.Row = 6; app.DLOpacitySlider.Layout.Column = [1, 2];

            lblSum = uilabel(app.DLCtrlGrid, 'Text', 'Distribusi Kelas (%)', ...
                'FontSize', 10, 'FontWeight', 'bold', 'FontColor', app.ColorInk);
            lblSum.Layout.Row = 7; lblSum.Layout.Column = [1, 2];

            app.DLClassSummaryArea = uitextarea(app.DLCtrlGrid, ...
                'Editable', 'off', ...
                'FontName', 'Consolas', ...
                'FontSize', 9.5, ...
                'Value', {'Belum ada hasil inferensi.'});
            app.DLClassSummaryArea.Layout.Row = 8; app.DLClassSummaryArea.Layout.Column = [1, 2];

            lblMode = uilabel(app.DLCtrlGrid, 'Text', 'Mode: Offline (weights/best_model.pth)', ...
                'FontSize', 9, 'FontColor', app.ColorMuted);
            lblMode.Layout.Row = 9; lblMode.Layout.Column = [1, 2];

            % Panel Visualisasi Kanan (3 Axes + Legend Bar 11 Kelas + Status)
            app.DLViewPanel = uipanel(app.DLTabGrid);
            app.styleSurfacePanel(app.DLViewPanel);

            app.DLViewGrid = uigridlayout(app.DLViewPanel, [3, 3]);
            app.DLViewGrid.RowHeight = {'1x', 82, 24};
            app.DLViewGrid.Padding = [12, 12, 12, 10];
            app.DLViewGrid.RowSpacing = 8;
            app.DLViewGrid.ColumnSpacing = 10;
            app.DLViewGrid.BackgroundColor = app.ColorSurface;

            app.DLAxOrig = uiaxes(app.DLViewGrid);
            app.DLAxOrig.Layout.Row = 1; app.DLAxOrig.Layout.Column = 1;

            app.DLAxColorMask = uiaxes(app.DLViewGrid);
            app.DLAxColorMask.Layout.Row = 1; app.DLAxColorMask.Layout.Column = 2;

            app.DLAxOverlay = uiaxes(app.DLViewGrid);
            app.DLAxOverlay.Layout.Row = 1; app.DLAxOverlay.Layout.Column = 3;

            app.DLAxLegend = uiaxes(app.DLViewGrid);
            app.DLAxLegend.Layout.Row = 2; app.DLAxLegend.Layout.Column = [1, 3];

            app.DLStatusLabel = uilabel(app.DLViewGrid, 'Text', 'Status: Siap.', ...
                'FontSize', 9.5, 'FontColor', app.ColorMuted);
            app.DLStatusLabel.Layout.Row = 3; app.DLStatusLabel.Layout.Column = [1, 3];
        end

    end

    methods (Access = public)

        function app = SegmentationApp
            % Constructor: Bangun komponen UI dan jalankan startupFcn
            createComponents(app);
            registerApp(app, app.UIFigure);
            runStartupFcn(app, @startupFcn);

            if nargout == 0
                clear app
            end
        end

        function delete(app)
            % Destructor: Tutup figure utama saat aplikasi dihapus
            if isvalid(app.UIFigure)
                delete(app.UIFigure);
            end
        end

    end
end
