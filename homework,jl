using LinearAlgebra
using Random
using Printf
using Plots

function gr_sh(A::Matrix{Float64})
    m, n = size(A)
    Q = similar(A)
    R = zeros(n, n)
    
    for j = 1:n
        v = copy(A[:, j])  
        
        for i = 1:j-1
            R[i, j] = 0.0
            for k = 1:m
                R[i, j] += Q[k, i] * v[k] 
            end

            for k = 1:m
                v[k] -= R[i, j] * Q[k, i]
            end
        end
        
        R[j, j] = 0.0
        for k = 1:m
            R[j, j] += v[k] * v[k]
        end
        R[j, j] = sqrt(R[j, j])
        
        for k = 1:m
            Q[k, j] = v[k] / R[j, j]
        end
    end
    
    return Q, R
end

function gv_rot(A::Matrix{Float64})
    m, n = size(A)
    R = copy(A)
    Q = zeros(m, m)
    for i = 1:m
        for j = 1:m
            if i == j
                Q[i, j] = 1.0
            else 
                Q[i, j] = 0.0
            end
        end
    end
    for j = 1:n
        for i = j+1:m
            a = R[j, j]
            b = R[i, j]
            r = sqrt(a*a + b*b)
            if r > 0
                c = a / r
                s = -b / r
                for k = j:n
                    r_jk = R[j, k]
                    r_ik = R[i, k]
                    R[j, k] = c * r_jk - s * r_ik
                    R[i, k] = s * r_jk + c * r_ik
                end
                for k = 1:m
                    q_kj = Q[k, j]
                    q_ki = Q[k, i]
                    Q[k, j] = c * q_kj - s * q_ki
                    Q[k, i] = s * q_kj + c * q_ki
                end
            end
        end
    end
    return Q, R
end

function hh_qr(A::Matrix{Float64})
    m, n = size(A)
    R = copy(A)
    V = []
    βs = []
    for k = 1:n
        x = zeros(m - k + 1)
        for i = 1:(m - k + 1)
            x[i] = R[k + i - 1, k]
        end
        sum_squares = 0.0
        for i = 1:length(x)
            sum_squares = sum_squares + x[i] * x[i]
        end
        norm_x = sqrt(sum_squares)
        v = copy(x)
        if x[1] >= 0
            v[1] = v[1] + norm_x
        else
            v[1] = v[1] - norm_x
        end
        sum_squares_v = 0.0
        for i = 1:length(v)
            sum_squares_v = sum_squares_v + v[i] * v[i]
        end
        β = 2.0 / sum_squares_v
        n_cols = n - k + 1
        vT_R = zeros(n_cols)
        for j = 1:n_cols
            sum_val = 0.0
            for i = 1:length(v)
                sum_val = sum_val + v[i] * R[k + i - 1, k + j - 1]
            end
            vT_R[j] = sum_val
        end
        for i = 1:length(v)
            for j = 1:n_cols
                R[k + i - 1, k + j - 1] = R[k + i - 1, k + j - 1] - β * v[i] * vT_R[j]
            end
        end
        push!(V, v)
        push!(βs, β)
    end
    Q = zeros(m, m)
    for i = 1:m
        for j = 1:m
            if i == j
                Q[i, j] = 1.0
            else
                Q[i, j] = 0.0
            end
        end
    end
    for k = n:-1:1
        v = V[k]
        β = βs[k]
        vT_Q = zeros(m)
        for j = 1:m
            sum_val = 0.0
            for i = 1:length(v)
                sum_val = sum_val + v[i] * Q[k + i - 1, j]
            end
            vT_Q[j] = sum_val
        end
        for i = 1:length(v)
            for j = 1:m
                Q[k + i - 1, j] = Q[k + i - 1, j] - β * v[i] * vT_Q[j]
            end
        end
    end
    return Q, R
end

function ill_conditioned_matrix(n, condition_number)
    U = qr(randn(n, n)).Q
    V = qr(randn(n, n)).Q
    Σ = diagm(LinRange(1.0, 1.0/condition_number, n))
    return U * Σ * V'
end

function orthogonal_error(Q::Matrix{Float64})
    m, n = size(Q)
    QtQ = zeros(n, n)
    for i = 1:n
        for j = 1:n
            sum_val = 0.0
            for k = 1:m
                sum_val = sum_val + Q[k, i] * Q[k, j]
            end
            QtQ[i, j] = sum_val
        end
    end
    diff = zeros(n, n)
    for i = 1:n
        for j = 1:n
            if i == j
                diff[i, j] = QtQ[i, j] - 1.0
            else
                diff[i, j] = QtQ[i, j]
            end
        end
    end
    max_abs = 0.0
    for i = 1:n
        for j = 1:n
            abs_val = abs(diff[i, j])
            if abs_val > max_abs
                max_abs = abs_val
            end
        end
    end
    return max_abs
end

function decomposition_error(A::Matrix{Float64}, Q::Matrix{Float64}, R::Matrix{Float64})
    m, n = size(A)
    QR = zeros(m, n)
    for i = 1:m
        for j = 1:n
            sum_val = 0.0
            for k = 1:n
                sum_val = sum_val + Q[i, k] * R[k, j]
            end
            QR[i, j] = sum_val
        end
    end
    diff = zeros(m, n)
    for i = 1:m
        for j = 1:n
            diff[i, j] = A[i, j] - QR[i, j]
        end
    end
    max_abs = 0.0
    for i = 1:m
        for j = 1:n
            abs_val = abs(diff[i, j])
            if abs_val > max_abs
                max_abs = abs_val
            end
        end
    end
    return max_abs
end

function run_qr_experiments()
    println("="^60)
    println("ЭКСПЕРИМЕНТАЛЬНОЕ ИССЛЕДОВАНИЕ QR-РАЗЛОЖЕНИЯ")
    println("="^60)
    
    sizes = [10, 20, 50, 100, 200, 500]
    condition_numbers = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    
    results_random = Dict()
    results_hilbert = Dict()
    results_condition = Dict()
    
    for method_name in ["gr_sh", "gv_rot", "hh_qr"]
        results_random[method_name] = Dict(
            "times" => Float64[],
            "orth_errors" => Float64[],
            "decomp_errors" => Float64[]
        )
        
        results_hilbert[method_name] = Dict(
            "times" => Float64[],
            "orth_errors" => Float64[],
            "decomp_errors" => Float64[]
        )
        
        for n in sizes
            if !haskey(results_condition, n)
                results_condition[n] = Dict()
            end
            results_condition[n][method_name] = Dict(
                "orth_errors" => Float64[]
            )
        end
    end
    
    println("\nЭКСПЕРИМЕНТ 1: Случайные матрицы")
    println("="^50)
    
    for n in sizes
        println("\nРазмер матрицы: $(n)×$(n)")
        println("-"^30)
        Random.seed!(123 + n)
        A = randn(n, n)
        
        for (method_name, method_func) in [
            ("gr_sh", gr_sh),
            ("gv_rot", gv_rot),
            ("hh_qr", hh_qr)
        ]
            start_time_ns = time_ns()  
            Q, R = method_func(A)
            elapsed_time = (time_ns() - start_time_ns) / 1e9  
            orth_err = orthogonal_error(Q)
            decomp_err = decomposition_error(A, Q, R)
            
            push!(results_random[method_name]["times"], elapsed_time)
            push!(results_random[method_name]["orth_errors"], orth_err)
            push!(results_random[method_name]["decomp_errors"], decomp_err)
            println(@sprintf("  %-12s: время = %9.6f с, ‖QᵀQ-I‖ = %8.2e", 
                           method_name, elapsed_time, orth_err))
        end
    end
    
    println("\n\nЭКСПЕРИМЕНТ 2: Матрицы Гильберта")
    println("="^70)
    
    for n in sizes
        println("\nРазмер матрицы: $(n)×$(n)")
        println("-"^30)
        
        A = zeros(n, n)
        for i = 1:n
            for j = 1:n
                A[i, j] = 1.0 / (i + j - 1)
            end
        end
        
        for (method_name, method_func) in [
            ("gr_sh", gr_sh),
            ("gv_rot", gv_rot),
            ("hh_qr", hh_qr)
        ]
            start_time_ns = time_ns()
            Q, R = method_func(A)
            elapsed_time = (time_ns() - start_time_ns) / 1e9
            
            orth_err = orthogonal_error(Q)
            decomp_err = decomposition_error(A, Q, R)
            
            push!(results_hilbert[method_name]["times"], elapsed_time)
            push!(results_hilbert[method_name]["orth_errors"], orth_err)
            push!(results_hilbert[method_name]["decomp_errors"], decomp_err)
            
            println(@sprintf("  %-12s: время = %9.6f с, ‖QᵀQ-I‖ = %8.2e, ‖A-QR‖ = %8.2e",
                           method_name, elapsed_time, orth_err, decomp_err))
        end
    end
    
    println("\n\nЭКСПЕРИМЕНТ 3: Зависимость от числа обусловленности")
    println("="^70)
    sizes_for_condition = [10, 20, 50, 100, 200, 500]  
    
    for n in sizes_for_condition
        println("\n\nРАЗМЕР МАТРИЦЫ: $(n)×$(n)")
        println("="^40)
        
        for (idx, cond_num) in enumerate(condition_numbers)
            println("\nЧисло обусловленности: $(cond_num)")
            println("-"^30)
            Random.seed!(123 + n + idx*1000)
            A = ill_conditioned_matrix(n, cond_num)
            
            for (method_name, method_func) in [
                ("gr_sh", gr_sh),
                ("gv_rot", gv_rot),
                ("hh_qr", hh_qr)
            ]
                Q, R = method_func(A)
                orth_err = orthogonal_error(Q)
                push!(results_condition[n][method_name]["orth_errors"], orth_err)
                
                println(@sprintf("  %-12s: ‖QᵀQ-I‖ = %8.2e",
                               method_name, orth_err))
            end
        end
    end
    
    return results_random, results_hilbert, results_condition, sizes, condition_numbers
end

function analyze_computational_complexity(results_random, results_hilbert, sizes)
    println("\n" * "="^60)
    println("АНАЛИЗ ВЫЧИСЛИТЕЛЬНОЙ СЛОЖНОСТИ")
    println("="^60)
    println("\n1. ТАБЛИЦА ВРЕМЕНИ ВЫПОЛНЕНИЯ (секунды):")
    println("   " * "-"^65)
    println("   Размер | Грамм-Шмидт   Гивенс       Хаусхолдер")
    println("   " * "-"^65)

    for (idx, n) in enumerate(sizes)
        t_gr_rand = results_random["gr_sh"]["times"][idx]
        t_gv_rand = results_random["gv_rot"]["times"][idx]
        t_hh_rand = results_random["hh_qr"]["times"][idx]
        
        println(@sprintf("   %6d | %10.4f  %10.4f  %10.4f  (случайные)", 
                       n, t_gr_rand, t_gv_rand, t_hh_rand))
        
        t_gr_hilb = results_hilbert["gr_sh"]["times"][idx]
        t_gv_hilb = results_hilbert["gv_rot"]["times"][idx]
        t_hh_hilb = results_hilbert["hh_qr"]["times"][idx]
        
        println(@sprintf("         | %10.4f  %10.4f  %10.4f  (Гильберт)", 
                       t_gr_hilb, t_gv_hilb, t_hh_hilb))
        println("   " * "-"^65)
    end
    println("\n\n2. АНАЛИЗ АСИМПТОТИЧЕСКОГО ПОВЕДЕНИЯ:")
    println("   " * "-"^50)
    println("   Оценка порядка сложности O(n³):")
    println("   (отношение времени для n=500 к теоретическому)")
    for method in ["gr_sh", "gv_rot", "hh_qr"]
        t10_rand = results_random[method]["times"][1]
        C_rand = t10_rand / (10^3)
        t500_pred_rand = C_rand * (500^3)
        t500_real_rand = results_random[method]["times"][end]
        ratio_rand = t500_real_rand / t500_pred_rand
        t10_hilb = results_hilbert[method]["times"][1]
        C_hilb = t10_hilb / (10^3)
        t500_pred_hilb = C_hilb * (500^3)
        t500_real_hilb = results_hilbert[method]["times"][end]
        ratio_hilb = t500_real_hilb / t500_pred_hilb
        
        method_name = method == "gr_sh" ? "Грамм-Шмидт" :
                     method == "gv_rot" ? "Гивенс" : "Хаусхолдер"
        
        println(@sprintf("   %-14s: случайные=%.2f, Гильберт=%.2f", 
                       method_name, ratio_rand, ratio_hilb))
    end
    
    println("\n\n3. ОТНОСИТЕЛЬНАЯ СКОРОСТЬ МЕТОДОВ (n=500):")
    println("   " * "-"^50)
    t_gr_500_rand = results_random["gr_sh"]["times"][end]
    t_gv_500_rand = results_random["gv_rot"]["times"][end]
    t_hh_500_rand = results_random["hh_qr"]["times"][end]
    min_time_rand = min(t_gr_500_rand, t_gv_500_rand, t_hh_500_rand)
    
    t_gr_500_hilb = results_hilbert["gr_sh"]["times"][end]
    t_gv_500_hilb = results_hilbert["gv_rot"]["times"][end]
    t_hh_500_hilb = results_hilbert["hh_qr"]["times"][end]
    min_time_hilb = min(t_gr_500_hilb, t_gv_500_hilb, t_hh_500_hilb)
    
    println("\n   Случайные матрицы:")
    println("   " * "-"^30)
    println(@sprintf("   Грамм-Шмидт:  %.4f с (%.1f×)", t_gr_500_rand, t_gr_500_rand/min_time_rand))
    println(@sprintf("   Гивенс:       %.4f с (%.1f×)", t_gv_500_rand, t_gv_500_rand/min_time_rand))
    println(@sprintf("   Хаусхолдер:   %.4f с (база)", t_hh_500_rand))
    
    println("\n   Матрицы Гильберта:")
    println("   " * "-"^30)
    println(@sprintf("   Грамм-Шмидт:  %.4f с (%.1f×)", t_gr_500_hilb, t_gr_500_hilb/min_time_hilb))
    println(@sprintf("   Гивенс:       %.4f с (%.1f×)", t_gv_500_hilb, t_gv_500_hilb/min_time_hilb))
    println(@sprintf("   Хаусхолдер:   %.4f с (база)", t_hh_500_hilb))
end

function analyze_numerical_stability(results_hilbert, results_condition, sizes, condition_numbers)
    println("\n" * "="^60)
    println("АНАЛИЗ ЧИСЛЕННОЙ УСТОЙЧИВОСТИ")
    println("="^60)
    println("\n1. ОШИБКИ ОРТОГОНАЛЬНОСТИ НА МАТРИЦАХ ГИЛЬБЕРТА:")
    println("   " * "-"^60)
    println("   Размер | Грамм-Шмидт     Гивенс       Хаусхолдер")
    println("   " * "-"^60)
    for (idx, n) in enumerate(sizes)
        e_gr = results_hilbert["gr_sh"]["orth_errors"][idx]
        e_gv = results_hilbert["gv_rot"]["orth_errors"][idx]
        e_hh = results_hilbert["hh_qr"]["orth_errors"][idx]
        println(@sprintf("   %6d | %12.2e  %12.2e  %12.2e", n, e_gr, e_gv, e_hh))
    end
    println("\n\n2. РОСТ ОШИБОК ПРИ УВЕЛИЧЕНИИ РАЗМЕРА:")
    println("   " * "-"^50)
    for method in ["gr_sh", "gv_rot", "hh_qr"]
        e10 = results_hilbert[method]["orth_errors"][1]
        e500 = results_hilbert[method]["orth_errors"][end]
        error_growth = e500 / e10
        method_name = method == "gr_sh" ? "Грамм-Шмидт" :
                     method == "gv_rot" ? "Гивенс" : "Хаусхолдер"
        if error_growth > 1e6
            growth_desc = "КАТАСТРОФИЧЕСКИЙ РОСТ"
        elseif error_growth > 1e3
            growth_desc = "БЫСТРЫЙ РОСТ"
        elseif error_growth > 10
            growth_desc = "УМЕРЕННЫЙ РОСТ"
        else
            growth_desc = "МЕДЛЕННЫЙ РОСТ"
        end
        println(@sprintf("   %-14s: рост в %.1e раз (%s)", 
                       method_name, error_growth, growth_desc))
    end
    println("\n\n3. ЗАВИСИМОСТЬ ОТ ЧИСЛА ОБУСЛОВЛЕННОСТИ:")
    println("   (чем больше ошибка, тем хуже устойчивость)")
    key_sizes = [10, 20, 50, 100, 200, 500]
    for n in key_sizes
        if haskey(results_condition, n)
            println("\n   Размер матрицы: $(n)×$(n)")
            println("   " * "-"^40)
            println("   Число об. | Грамм-Шмидт   Гивенс       Хаусхолдер")
            println("   " * "-"^40)
            
            for (idx, cond_num) in enumerate(condition_numbers)
                e_gr = results_condition[n]["gr_sh"]["orth_errors"][idx]
                e_gv = results_condition[n]["gv_rot"]["orth_errors"][idx]
                e_hh = results_condition[n]["hh_qr"]["orth_errors"][idx]
                
                println(@sprintf("   %8.0e | %12.2e  %12.2e  %12.2e", 
                               cond_num, e_gr, e_gv, e_hh))
            end
        end
    end
end

function plot_experiment_results(results_random, results_hilbert, results_condition, sizes, condition_numbers)
    println("\n" * "="^60)
    println("ПОСТРОЕНИЕ ГРАФИКОВ")
    println("="^60)
    plot_layout = @layout [a b c; d e f; g h i; j]
    p = plot(layout=plot_layout, size=(1400, 1600))

    plot!(p[1], 
          title="Время выполнения (случайные матрицы)",
          xlabel="Размер матрицы n", 
          ylabel="Время (с)",
          xscale=:log10,
          yscale=:log10,
          legend=:topleft,
          xticks=(sizes, string.(sizes)))  
    
    plot!(p[1], sizes, results_random["gr_sh"]["times"],
          label="Грамм-Шмидт", marker=:circle, linewidth=2)
    plot!(p[1], sizes, results_random["gv_rot"]["times"],
          label="Гивенс", marker=:square, linewidth=2)
    plot!(p[1], sizes, results_random["hh_qr"]["times"],
          label="Хаусхолдер", marker=:diamond, linewidth=2)
    
    plot!(p[2], 
          title="Ошибка ортогональности (случайные)",
          xlabel="Размер матрицы n", 
          ylabel="‖QᵀQ - I‖",
          xscale=:log10,
          yscale=:log10,
          legend=:topleft,
          xticks=(sizes, string.(sizes)))  
    
    plot!(p[2], sizes, results_random["gr_sh"]["orth_errors"],
          label="Грамм-Шмидт", marker=:circle, linewidth=2)
    plot!(p[2], sizes, results_random["gv_rot"]["orth_errors"],
          label="Гивенс", marker=:square, linewidth=2)
    plot!(p[2], sizes, results_random["hh_qr"]["orth_errors"],
          label="Хаусхолдер", marker=:diamond, linewidth=2)
    
    plot!(p[3], 
          title="Ошибка ортогональности (Гильберт)",
          xlabel="Размер матрицы n", 
          ylabel="‖QᵀQ - I‖",
          xscale=:log10,
          yscale=:log10,
          legend=:topleft,
          xticks=(sizes, string.(sizes)))  
    
    plot!(p[3], sizes, results_hilbert["gr_sh"]["orth_errors"],
          label="Грамм-Шмидт", marker=:circle, linewidth=2)
    plot!(p[3], sizes, results_hilbert["gv_rot"]["orth_errors"],
          label="Гивенс", marker=:square, linewidth=2)
    plot!(p[3], sizes, results_hilbert["hh_qr"]["orth_errors"],
          label="Хаусхолдер", marker=:diamond, linewidth=2)
    key_sizes = [10, 20, 50, 100, 200, 500]
    plot_indices = [4, 5, 6, 7, 8, 9]
    for (i, n) in enumerate(key_sizes)
        if haskey(results_condition, n)
            plot_idx = plot_indices[i]
            
            plot!(p[plot_idx],
                  title="Зависимость от об. (n=$n)",
                  xlabel="Число обусловленности",
                  ylabel="‖QᵀQ - I‖",
                  xaxis=:log10,
                  yaxis=:log10,
                  legend=(i==1) ? :topleft : false)
            
            plot!(p[plot_idx], condition_numbers, results_condition[n]["gr_sh"]["orth_errors"],
                  label="Грамм-Шмидт", marker=:circle, linewidth=2)
            plot!(p[plot_idx], condition_numbers, results_condition[n]["gv_rot"]["orth_errors"],
                  label="Гивенс", marker=:square, linewidth=2)
            plot!(p[plot_idx], condition_numbers, results_condition[n]["hh_qr"]["orth_errors"],
                  label="Хаусхолдер", marker=:diamond, linewidth=2)
        end
    end
    
    plot!(p[10],
          title="Время выполнения (матрицы Гильберта)",
          xlabel="Размер матрицы n",
          ylabel="Время (с)",
          xaxis=:log10,
          yaxis=:log10,
          legend=:topleft)
    plot!(p[10], sizes, results_hilbert["gr_sh"]["times"],
          label="Грамм-Шмидт", marker=:circle, linewidth=2)
    plot!(p[10], sizes, results_hilbert["gv_rot"]["times"],
          label="Гивенс", marker=:square, linewidth=2)
    plot!(p[10], sizes, results_hilbert["hh_qr"]["times"],
          label="Хаусхолдер", marker=:diamond, linewidth=2)
    display(p)
    savefig(p, "qr_comparison_results.png")
    println("\nГрафики сохранены в файл: qr_comparison_results.png")
    return p
end

function main()
    println("\n" * "="^80)
    println("ПОЛНОЕ ИССЛЕДОВАНИЕ QR-РАЗЛОЖЕНИЙ")
    println("Размеры матриц: 10, 20, 50, 100, 200, 500")
    println("="^80)
    println("\nШАГ 1: Проведение экспериментов...")
    results_random, results_hilbert, results_condition, sizes, condition_numbers = run_qr_experiments()
    println("\nШАГ 2: Анализ вычислительной сложности...")
    analyze_computational_complexity(results_random, results_hilbert, sizes)
    println("\nШАГ 3: Анализ численной устойчивости...")
    analyze_numerical_stability(results_hilbert, results_condition, sizes, condition_numbers)
    println("\nШАГ 4: Построение графиков...")
    plot_experiment_results(results_random, results_hilbert, results_condition, sizes, condition_numbers)
    return results_random, results_hilbert, results_condition
    println("Конец")
end

main()
