classdef gui_helpers
    % GUI_HELPERS Kumpulan fungsi bantu terpusat untuk GUI SegmentationApp.
    %
    % Memisahkan seluruh logika resolusi path, jembatan Python (offline),
    % alpha-blending lokal, visualisasi legend CamVid 11 kelas, serta
    % pemanggilan + stub sementara modul Anggota A agar kode GUI tetap ringkas
    % dan mudah di-review di Git.

    methods (Static)

        function paths = resolve_project_paths()
            % RESOLVE_PROJECT_PATHS Menentukan seluruh path proyek secara relatif
            % berbasis lokasi file ini (tanpa hardcoded path absolut).

            guiDir = fileparts(mfilename('fullpath'));
            projectRootRaw = fullfile(guiDir, '..', '..');
            projectRoot = gui_helpers.canonicalize_path(projectRootRaw);

            % Tambahkan folder GUI serta folder modul Anggota A (bila tersedia) ke MATLAB path
            addpath(guiDir);
            edgeDir = fullfile(projectRoot, 'src', 'edge_segmentation');
            splitDir = fullfile(projectRoot, 'src', 'split_merge');
            if isfolder(edgeDir)
                addpath(edgeDir);
            end
            if isfolder(splitDir)
                addpath(splitDir);
            end

            % Prioritaskan interpreter Python di dalam virtual environment proyek
            winPy = fullfile(projectRoot, 'venv', 'Scripts', 'python.exe');
            posixPy = fullfile(projectRoot, 'venv', 'bin', 'python');
            if isfile(winPy)
                pythonExe = winPy;
            elseif isfile(posixPy)
                pythonExe = posixPy;
            else
                pythonExe = 'python';
            end

            paths = struct();
            paths.guiDir = guiDir;
            paths.projectRoot = projectRoot;
            paths.pythonExe = pythonExe;
            paths.inferScript = fullfile(projectRoot, 'src', 'deep_learning', 'infer_segmentation.py');
            paths.weightsPath = fullfile(projectRoot, 'src', 'deep_learning', 'weights', 'best_model.pth');
            paths.bsdsDir = fullfile(projectRoot, 'data', 'bsds500', 'bsds500_test_8');
            paths.camvidTestDir = fullfile(projectRoot, 'data', 'camvid', 'test');
            paths.cacheDir = fullfile(projectRoot, 'src', 'deep_learning', 'outputs', 'gui_cache');
        end

        function outPath = canonicalize_path(inPath)
            % CANONICALIZE_PATH Menormalkan segmen '..' menjadi path absolut rapi.
            outPath = char(inPath);
            try
                outPath = char(java.io.File(outPath).getCanonicalPath());
            catch
                % Pertahankan path bawaan jika JVM tidak aktif
            end
        end

        function imgOut = normalize_image_uint8(imgIn)
            % NORMALIZE_IMAGE_UINT8 Menyeragamkan citra ke tipe uint8 (1 atau 3 kanal)
            % agar aman ditampilkan di UIAxes dan diproses oleh modul backend.

            if isempty(imgIn)
                error('gui_helpers:EmptyImage', 'Citra masukan kosong.');
            end

            % Buang kanal alpha jika citra berupa RGBA (4 kanal)
            if ndims(imgIn) == 3 && size(imgIn, 3) > 3
                imgIn = imgIn(:, :, 1:3);
            end

            if islogical(imgIn)
                imgOut = uint8(imgIn) * 255;
            elseif isa(imgIn, 'uint8')
                imgOut = imgIn;
            elseif isa(imgIn, 'uint16')
                imgOut = uint8(double(imgIn) / 257.0);
            elseif isfloat(imgIn)
                maxVal = max(imgIn(:));
                if maxVal <= 1.0
                    imgOut = uint8(round(min(max(imgIn, 0), 1) * 255));
                else
                    imgOut = uint8(round(min(max(imgIn, 0), 255)));
                end
            else
                imgOut = uint8(imgIn);
            end
        end

        function rgb = to_rgb_uint8(imgIn)
            % TO_RGB_UINT8 Mengonversi citra grayscale maupun RGB menjadi uint8 (H x W x 3).
            u8 = gui_helpers.normalize_image_uint8(imgIn);
            if ndims(u8) == 2 || size(u8, 3) == 1
                rgb = repmat(u8(:, :, 1), [1, 1, 3]);
            else
                rgb = u8(:, :, 1:3);
            end
        end

        function result = run_dl_segmentation(imgOrPath, imgName, opacity, paths)
            % RUN_DL_SEGMENTATION Menjalankan inferensi semantic segmentation secara offline
            % melalui skrip infer_segmentation.py menggunakan Python dari venv proyek.
            %
            % Output struct `result`:
            %   - colorMask        : (uint8, H x W x 3) Peta warna 11 kelas CamVid
            %   - overlayImg       : (uint8, H x W x 3) Gabungan citra asli + colorMask
            %   - labelMask        : (uint8, H x W) Indeks kelas 0..10
            %   - legendImg        : (uint8) Gambar legend 11 kelas dari skrip Python
            %   - legendMeta       : Struct metadata 11 kelas (id, name, rgb)
            %   - classPercentages : Struct persentase piksel per kelas
            %   - inputSizeHW      : [H, W] ukuran asli citra

            if nargin < 4 || isempty(paths)
                paths = gui_helpers.resolve_project_paths();
            end
            if nargin < 3 || isempty(opacity)
                opacity = 0.5;
            end
            if nargin < 2 || isempty(imgName)
                imgName = 'input_image';
            end

            % Validasi kesiapan skrip, bobot lokal, dan interpreter Python venv
            if ~isfile(paths.inferScript)
                error('gui_helpers:MissingScript', ...
                    'Skrip inferensi tidak ditemukan: %s', paths.inferScript);
            end
            if ~isfile(paths.weightsPath)
                error('gui_helpers:MissingWeights', ...
                    'File bobot model lokal tidak ditemukan: %s', paths.weightsPath);
            end
            if ~strcmp(paths.pythonExe, 'python') && ~isfile(paths.pythonExe)
                error('gui_helpers:MissingPython', ...
                    'Python virtual environment tidak ditemukan: %s', paths.pythonExe);
            end

            if ~isfolder(paths.cacheDir)
                mkdir(paths.cacheDir);
            end

            % Tentukan path citra masukan yang dikirim ke CLI Python
            if (ischar(imgOrPath) || isstring(imgOrPath)) && isfile(imgOrPath)
                inputPath = char(imgOrPath);
                [~, stem, ~] = fileparts(inputPath);
            else
                [~, rawStem, ~] = fileparts(char(imgName));
                if isempty(rawStem)
                    rawStem = 'gui_input';
                end
                % Bersihkan karakter non-alfanumerik pada nama file cache
                stem = regexprep(rawStem, '[^a-zA-Z0-9_-]', '_');
                inputPath = fullfile(paths.cacheDir, [stem '_input.png']);
                imwrite(gui_helpers.to_rgb_uint8(imgOrPath), inputPath);
            end

            alpha = min(max(double(opacity), 0.0), 1.0);

            % Susun perintah CLI Python dengan path ter-quote agar aman terhadap spasi.
            % MATLAB system() sudah menangani pembungkus shell pada Windows, sehingga
            % tidak perlu membungkus ulang keseluruhan perintah dengan kutip ganda.
            cmd = sprintf('"%s" "%s" "%s" --output-dir "%s" --weights "%s" --opacity %.4f --device auto', ...
                paths.pythonExe, paths.inferScript, inputPath, paths.cacheDir, paths.weightsPath, alpha);

            [status, cmdOut] = system(cmd);
            if status ~= 0
                error('gui_helpers:PythonInferenceFailed', ...
                    'Inferensi Python gagal (exit code %d).\nDetail:\n%s', status, cmdOut);
            end

            % Baca kembali seluruh artefak keluaran infer_segmentation.py
            colorPath = fullfile(paths.cacheDir, [stem '_color.png']);
            overlayPath = fullfile(paths.cacheDir, [stem '_overlay.png']);
            maskPath = fullfile(paths.cacheDir, [stem '_mask.png']);
            legendPath = fullfile(paths.cacheDir, 'legend_11_classes.png');
            metaPath = fullfile(paths.cacheDir, [stem '_meta.json']);

            if ~isfile(colorPath) || ~isfile(overlayPath)
                error('gui_helpers:MissingOutputs', ...
                    'Output inferensi tidak lengkap di folder cache: %s', paths.cacheDir);
            end

            result = struct();
            result.colorMask = imread(colorPath);
            result.overlayImg = imread(overlayPath);
            result.labelMask = [];
            if isfile(maskPath)
                result.labelMask = imread(maskPath);
            end
            result.legendImg = [];
            if isfile(legendPath)
                result.legendImg = imread(legendPath);
            end

            result.legendMeta = gui_helpers.get_camvid_classes();
            result.classPercentages = struct();
            result.inputSizeHW = [size(result.colorMask, 1), size(result.colorMask, 2)];

            if isfile(metaPath)
                metaRaw = fileread(metaPath);
                meta = jsondecode(metaRaw);
                if isfield(meta, 'class_percentages')
                    result.classPercentages = meta.class_percentages;
                end
                if isfield(meta, 'input_size_hw')
                    result.inputSizeHW = meta.input_size_hw;
                end
            end
        end

        function overlayImg = blend_overlay(origImg, colorMask, opacity)
            % BLEND_OVERLAY Melakukan alpha-blending lokal di MATLAB agar slider
            % opacity pada Tab 3 responsif secara real-time tanpa memanggil ulang CNN.
            %
            % Formula identik dengan create_overlay() di infer_segmentation.py:
            %   overlay = (1 - alpha) * origRGB + alpha * colorMask

            origRGB = gui_helpers.to_rgb_uint8(origImg);
            maskRGB = gui_helpers.to_rgb_uint8(colorMask);

            % Pastikan dimensi mask sama persis dengan citra asli (nearest untuk diskret warna)
            if size(maskRGB, 1) ~= size(origRGB, 1) || size(maskRGB, 2) ~= size(origRGB, 2)
                maskRGB = imresize(maskRGB, [size(origRGB, 1), size(origRGB, 2)], 'nearest');
            end

            alpha = min(max(double(opacity), 0.0), 1.0);
            blended = (1.0 - alpha) * double(origRGB) + alpha * double(maskRGB);
            overlayImg = uint8(min(max(round(blended), 0), 255));
        end

        function classes = get_camvid_classes()
            % GET_CAMVID_CLASSES Mengembalikan nama dan warna RGB 11 kelas CamVid
            % sesuai pemetaan di src/deep_learning/camvid_mapping.py.

            names = { ...
                'Sky', 'Building', 'Pole', 'Road', 'Pavement', ...
                'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist' ...
            };
            palette = uint8([ ...
                128, 128, 128; ... % 0: Sky
                128,   0,   0; ... % 1: Building
                192, 192, 128; ... % 2: Pole
                128,  64, 128; ... % 3: Road
                  0,   0, 192; ... % 4: Pavement
                128, 128,   0; ... % 5: Tree
                192, 128, 128; ... % 6: SignSymbol
                 64,  64, 128; ... % 7: Fence
                 64,   0, 128; ... % 8: Car
                 64,  64,   0; ... % 9: Pedestrian
                  0, 128, 192  ... % 10: Bicyclist
            ]);

            classes = repmat(struct('id', 0, 'name', '', 'rgb', uint8([0 0 0])), 1, numel(names));
            for i = 1:numel(names)
                classes(i).id = i - 1;
                classes(i).name = names{i};
                classes(i).rgb = palette(i, :);
            end
        end

        function render_legend(ax, classPercentages)
            % RENDER_LEGEND Menggambar legend warna 11 kelas CamVid pada UIAxes Tab 3
            % agar selalu terlihat rapi tanpa pop-up figure eksternal.

            if nargin < 2
                classPercentages = struct();
            end

            classes = gui_helpers.get_camvid_classes();
            n = numel(classes);

            cla(ax);
            hold(ax, 'on');
            for i = 1:n
                rgbNorm = double(classes(i).rgb) / 255.0;
                x0 = i - 1;
                rectangle(ax, 'Position', [x0 + 0.04, 0.12, 0.92, 0.76], ...
                    'FaceColor', rgbNorm, ...
                    'EdgeColor', [0.85, 0.87, 0.90], ...
                    'LineWidth', 0.8, ...
                    'Curvature', 0.12);

                % Hitung luminansi relatif untuk memilih teks putih atau gelap yang kontras
                lum = 0.299 * double(classes(i).rgb(1)) + ...
                      0.587 * double(classes(i).rgb(2)) + ...
                      0.114 * double(classes(i).rgb(3));
                if lum > 150
                    txtColor = [0.09, 0.11, 0.14];
                else
                    txtColor = [1.0, 1.0, 1.0];
                end

                labelStr = classes(i).name;
                if isstruct(classPercentages) && isfield(classPercentages, labelStr)
                    pct = classPercentages.(labelStr);
                    labelStr = sprintf('%s\n%.1f%%', labelStr, pct);
                end

                text(ax, x0 + 0.5, 0.5, labelStr, ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'FontSize', 8, ...
                    'FontWeight', 'bold', ...
                    'Color', txtColor);
            end
            hold(ax, 'off');

            xlim(ax, [0, n]);
            ylim(ax, [0, 1]);
            axis(ax, 'off');
        end

        function [edgeMap, objMask, segObjImg, isStub] = call_edge_segmentation(img, operator, params)
            % CALL_EDGE_SEGMENTATION Dispatcher untuk Modul 1 (Tab 1 GUI).
            % Otomatis memanggil `run_edge_segmentation` milik Anggota A apabila sudah
            % tersedia di MATLAB path, atau memakai stub sementara sesuai contract.md.

            if exist('run_edge_segmentation', 'file') == 2
                [edgeMap, objMask, segObjImg] = run_edge_segmentation(img, operator, params);
                isStub = false;
            else
                [edgeMap, objMask, segObjImg] = gui_helpers.stub_edge_segmentation(img, operator, params);
                isStub = true;
            end
        end

        function [splitVisImg, mergedImg, stats, isStub] = call_split_and_merge(img, splitThresh, mergeThresh, minBlock)
            % CALL_SPLIT_AND_MERGE Dispatcher untuk Modul 2 (Tab 2 GUI).
            % Otomatis memanggil `split_and_merge` milik Anggota A apabila sudah
            % tersedia di MATLAB path, atau memakai stub sementara sesuai contract.md.

            if exist('split_and_merge', 'file') == 2
                [splitVisImg, mergedImg, stats] = split_and_merge(img, splitThresh, mergeThresh, minBlock);
                isStub = false;
            else
                [splitVisImg, mergedImg, stats] = gui_helpers.stub_split_and_merge(img, splitThresh, mergeThresh, minBlock);
                isStub = true;
            end
        end

        function [edgeMap, objMask, segObjImg] = stub_edge_segmentation(img, operator, params)
            % STUB_EDGE_SEGMENTATION Implementasi placeholder Modul 1 sesuai contract.md
            % untuk pengujian layout & callback GUI sebelum kode Anggota A di-merge.

            if nargin < 2 || isempty(operator)
                operator = 'sobel';
            end
            if nargin < 3 || ~isstruct(params)
                params = struct();
            end
            if ~isfield(params, 'thresh') || isempty(params.thresh)
                params.thresh = 0.25;
            end
            if ~isfield(params, 'sigma') || isempty(params.sigma)
                params.sigma = 1.0;
            end
            if ~isfield(params, 'morphRadius') || isempty(params.morphRadius)
                params.morphRadius = 3;
            end
            if ~isfield(params, 'minArea') || isempty(params.minArea)
                params.minArea = 500;
            end

            imgU8 = gui_helpers.normalize_image_uint8(img);
            if ndims(imgU8) == 3 && size(imgU8, 3) == 3
                gray = double(rgb2gray(imgU8)) / 255.0;
            else
                gray = double(imgU8(:, :, 1)) / 255.0;
            end

            % Hitung gradien sederhana sebagai visualisasi placeholder responsif
            [gx, gy] = gradient(gray);
            mag = hypot(gx, gy);
            maxMag = max(mag(:));
            if maxMag > 0
                mag = mag / maxMag;
            end

            % Bedakan sedikit respons antar-operator agar perubahan dropdown terlihat saat uji GUI
            op = lower(strtrim(char(operator)));
            thresh = min(max(double(params.thresh), 0.01), 0.99);
            switch op
                case 'canny'
                    if exist('edge', 'file') == 2
                        edgeMap = edge(gray, 'canny', [], max(0.5, double(params.sigma)));
                    else
                        edgeMap = mag > thresh;
                    end
                case {'laplace', 'log'}
                    lap = del2(gray);
                    lapNorm = abs(lap);
                    if max(lapNorm(:)) > 0
                        lapNorm = lapNorm / max(lapNorm(:));
                    end
                    edgeMap = lapNorm > (thresh * 0.65);
                otherwise
                    edgeMap = mag > thresh;
            end

            % Bentuk mask objek placeholder dari peta tepi menggunakan operasi morfologi dasar
            radius = max(1, round(double(params.morphRadius)));
            if exist('imclose', 'file') == 2 && exist('strel', 'file') == 2
                se = strel('disk', radius);
                objMask = imclose(edgeMap, se);
                objMask = imfill(objMask, 'holes');
                if exist('bwareafilt', 'file') == 2 && any(objMask(:))
                    filtered = bwareafilt(objMask, [max(1, round(double(params.minArea))), inf]);
                    if any(filtered(:))
                        objMask = filtered;
                    end
                end
            else
                objMask = logical(edgeMap);
            end

            % Isolasi objek pada latar belakang gelap
            if ndims(imgU8) == 3 && size(imgU8, 3) == 3
                segObjImg = imgU8 .* uint8(repmat(objMask, [1, 1, 3]));
            else
                segObjImg = imgU8(:, :, 1) .* uint8(objMask);
            end
        end

        function [splitVisImg, mergedImg, stats] = stub_split_and_merge(img, splitThresh, mergeThresh, minBlock)
            % STUB_SPLIT_AND_MERGE Implementasi placeholder Modul 2 sesuai contract.md
            % untuk pengujian GUI Tab 2 sebelum kode Anggota A di-merge.

            if nargin < 2 || isempty(splitThresh)
                splitThresh = 25;
            end
            if nargin < 3 || isempty(mergeThresh)
                mergeThresh = 15;
            end
            if nargin < 4 || isempty(minBlock)
                minBlock = 8;
            end

            rgb = gui_helpers.to_rgb_uint8(img);
            [h, w, ~] = size(rgb);
            blk = max(4, round(double(minBlock)));

            % Visualisasi grid blok sederhana untuk tahap "After Split"
            step = max(blk * 2, round(blk * (1 + double(splitThresh) / 50.0)));
            splitVisImg = rgb;
            gridColor = uint8([29, 78, 216]);
            for r = 1:step:h
                splitVisImg(r, :, 1) = gridColor(1);
                splitVisImg(r, :, 2) = gridColor(2);
                splitVisImg(r, :, 3) = gridColor(3);
            end
            for c = 1:step:w
                splitVisImg(:, c, 1) = gridColor(1);
                splitVisImg(:, c, 2) = gridColor(2);
                splitVisImg(:, c, 3) = gridColor(3);
            end

            % Kuantisasi blok sebagai visualisasi placeholder tahap "After Merge"
            mergedImg = rgb;
            qStep = max(8, min(128, round(double(mergeThresh) * 1.5)));
            mergedImg = uint8(floor(double(mergedImg) / qStep) * qStep + qStep / 2);

            numCols = ceil(w / step);
            numRows = ceil(h / step);
            stats = struct();
            stats.numSplitRegions = numRows * numCols;
            stats.numMergedRegions = max(1, round(stats.numSplitRegions / max(1.5, double(mergeThresh) / 10.0)));
        end

    end
end
