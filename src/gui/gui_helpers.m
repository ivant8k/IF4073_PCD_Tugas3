classdef gui_helpers
    % GUI_HELPERS Kumpulan fungsi bantu terpusat untuk GUI SegmentationApp.
    %
    % Memisahkan seluruh logika resolusi path, jembatan Python (offline),
    % alpha-blending lokal, dan visualisasi legend CamVid 11 kelas agar
    % kode GUI tetap ringkas dan mudah di-review di Git.

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
            % dalam tata letak 3 baris (4, 4, 3 label) agar mudah dibaca.

            if nargin < 2
                classPercentages = struct();
            end

            classes = gui_helpers.get_camvid_classes();
            n = numel(classes);
            maxCols = 4;
            numRows = 3;
            rowCounts = [4, 4, 3];

            cla(ax);
            hold(ax, 'on');
            for i = 1:n
                % Petakan indeks 1..11 ke baris (4, 4, 3) dari atas ke bawah
                if i <= 4
                    rowIdx = 1;
                    colIdx = i - 1;
                elseif i <= 8
                    rowIdx = 2;
                    colIdx = i - 5;
                else
                    rowIdx = 3;
                    colIdx = i - 9;
                end

                % Pusatkan baris terakhir yang berisi 3 label agar seimbang
                xOffset = (maxCols - rowCounts(rowIdx)) * 0.5;
                x0 = xOffset + colIdx;
                y0 = numRows - rowIdx;

                rgbNorm = double(classes(i).rgb) / 255.0;
                rectangle(ax, 'Position', [x0 + 0.04, y0 + 0.12, 0.92, 0.76], ...
                    'FaceColor', rgbNorm, ...
                    'EdgeColor', [0.85, 0.87, 0.90], ...
                    'LineWidth', 0.9, ...
                    'Curvature', 0.14);

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
                    labelStr = sprintf('%s (%.1f%%)', labelStr, pct);
                end

                text(ax, x0 + 0.5, y0 + 0.5, labelStr, ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'FontSize', 11.5, ...
                    'FontWeight', 'bold', ...
                    'Color', txtColor);
            end
            hold(ax, 'off');

            xlim(ax, [0, maxCols]);
            ylim(ax, [0, numRows]);
            axis(ax, 'off');
        end

    end
end
