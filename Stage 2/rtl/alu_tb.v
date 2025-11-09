/*
 * File: alu_tb_enhanced.v
 * Description: Enhanced testbench for multiple Trojan variant detection
 * Author: Shivani Bhat 
 * Date: November 2025
 * 
 * This testbench:
 * - Tests clean ALU against 4 different Trojan variants
 * - Generates targeted trigger patterns for each variant
 * - Comprehensive VCD output for ML analysis
 * - Performance comparison across all variants
 */

`timescale 1ns/1ps

module alu_tb_enhanced;

    // Testbench signals
    reg clk;
    reg rst_n;
    reg [3:0] A;
    reg [3:0] B;
    reg [1:0] op;
    
    // Clean ALU outputs
    wire [3:0] result_clean;
    wire carry_out_clean;
    wire zero_flag_clean;
    
    // Trojan variant outputs
    wire [3:0] result_simple, result_seq, result_time, result_complex;
    wire carry_out_simple, carry_out_seq, carry_out_time, carry_out_complex;
    wire zero_flag_simple, zero_flag_seq, zero_flag_time, zero_flag_complex;
    
    // Test statistics
    integer test_count;
    integer mismatch_simple, mismatch_seq, mismatch_time, mismatch_complex;
    integer trigger_count_simple, trigger_count_seq, trigger_count_time, trigger_count_complex;
    
    integer i, j, k, r;
    
    // Instantiate Clean ALU
    alu_clean uut_clean (
        .clk(clk), .rst_n(rst_n),
        .A(A), .B(B), .op(op),
        .result(result_clean),
        .carry_out(carry_out_clean),
        .zero_flag(zero_flag_clean)
    );
    
    // Instantiate Trojan Variants
    alu_trojan_simple uut_simple (
        .clk(clk), .rst_n(rst_n),
        .A(A), .B(B), .op(op),
        .result(result_simple),
        .carry_out(carry_out_simple),
        .zero_flag(zero_flag_simple)
    );
    
    alu_trojan_sequential uut_seq (
        .clk(clk), .rst_n(rst_n),
        .A(A), .B(B), .op(op),
        .result(result_seq),
        .carry_out(carry_out_seq),
        .zero_flag(zero_flag_seq)
    );
    
    alu_trojan_timebased uut_time (
        .clk(clk), .rst_n(rst_n),
        .A(A), .B(B), .op(op),
        .result(result_time),
        .carry_out(carry_out_time),
        .zero_flag(zero_flag_time)
    );
    
    alu_trojan_complex uut_complex (
        .clk(clk), .rst_n(rst_n),
        .A(A), .B(B), .op(op),
        .result(result_complex),
        .carry_out(carry_out_complex),
        .zero_flag(zero_flag_complex)
    );
    
    // Clock generation: 10ns period
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // VCD dump
    initial begin
        $dumpfile("alu_variants_simulation.vcd");
        $dumpvars(0, alu_tb_enhanced);
        
        $dumpvars(1, uut_clean);
        $dumpvars(1, uut_simple);
        $dumpvars(1, uut_seq);
        $dumpvars(1, uut_time);
        $dumpvars(1, uut_complex);
    end
    
    // Main test stimulus
    initial begin
        // Initialize
        test_count = 0;
        mismatch_simple = 0;
        mismatch_seq = 0;
        mismatch_time = 0;
        mismatch_complex = 0;
        trigger_count_simple = 0;
        trigger_count_seq = 0;
        trigger_count_time = 0;
        trigger_count_complex = 0;
        
        rst_n = 0;
        A = 4'b0000;
        B = 4'b0000;
        op = 2'b00;
        
        // Display header
        $display("\n==============================================================");
        $display("ENHANCED HARDWARE TROJAN DETECTION TESTBENCH");
        $display("Testing Multiple Trojan Variants");
        $display("==============================================================\n");
        
        // Reset sequence
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // ============================================================
        // PHASE 1: Basic exhaustive testing
        // ============================================================
        $display("\n[PHASE 1] Exhaustive Input Testing (1024 tests)");
        $display("------------------------------------------------------------");
        
        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin
                for (k = 0; k < 4; k = k + 1) begin
                    A = i[3:0];
                    B = j[3:0];
                    op = k[1:0];
                    @(posedge clk);
                    #1;
                    check_all_results();
                end
            end
            
            if ((i % 4) == 0) begin
                $display("  Progress: %0d/16 patterns completed", i);
            end
        end
        $display("  [Phase 1 Complete] %0d tests executed\n", test_count);
        
        // ============================================================
        // PHASE 2: Simple Trojan triggers
        // ============================================================
        $display("\n[PHASE 2] Simple Trojan Trigger Testing");
        $display("------------------------------------------------------------");
        
        $display("  Testing: A=1111, B=1111, op=ADD (Trigger 1)");
        repeat(10) begin
            A = 4'b1111; B = 4'b1111; op = 2'b00;
            @(posedge clk); #1;
            check_all_results();
            trigger_count_simple = trigger_count_simple + 1;
        end
        
        $display("  Testing: A=0000, B=1111, op=AND (Trigger 2)");
        repeat(10) begin
            A = 4'b0000; B = 4'b1111; op = 2'b10;
            @(posedge clk); #1;
            check_all_results();
            trigger_count_simple = trigger_count_simple + 1;
        end
        $display("  [Phase 2 Complete]\n");
        
        // ============================================================
        // PHASE 3: Sequential Trojan triggers
        // ============================================================
        $display("\n[PHASE 3] Sequential Trojan Trigger Testing");
        $display("------------------------------------------------------------");
        $display("  Sending trigger sequence: Pattern1 -> Pattern2 -> Pattern3");
        
        repeat(5) begin
            // Pattern 1: ADD(1111,1111)
            A = 4'b1111; B = 4'b1111; op = 2'b00;
            @(posedge clk); #1; check_all_results();
            
            // Pattern 2: SUB(1000,0111)
            A = 4'b1000; B = 4'b0111; op = 2'b01;
            @(posedge clk); #1; check_all_results();
            
            // Pattern 3: AND(1010,0101)
            A = 4'b1010; B = 4'b0101; op = 2'b10;
            @(posedge clk); #1; check_all_results();
            trigger_count_seq = trigger_count_seq + 1;
            
            // Execute operations while Trojan is active
            repeat(8) begin
                A = $random; B = $random; op = $random;
                @(posedge clk); #1; check_all_results();
            end
        end
        $display("  [Phase 3 Complete]\n");
        
        // ============================================================
        // PHASE 4: Time-based Trojan triggers
        // ============================================================
        $display("\n[PHASE 4] Time-Based Trojan Trigger Testing");
        $display("------------------------------------------------------------");
        $display("  Arming time-based Trojan: OR(1100, 0011)");
        
        repeat(3) begin
            // Arm the Trojan
            A = 4'b1100; B = 4'b0011; op = 2'b11;
            @(posedge clk); #1; check_all_results();
            trigger_count_time = trigger_count_time + 1;
            
            // Wait for activation (256 cycles + 16 active cycles)
            $display("  Waiting for activation (272 cycles)...");
            repeat(280) begin
                A = $random; B = $random; op = $random;
                @(posedge clk); #1;
                if ((test_count % 50) == 0) check_all_results();
            end
        end
        $display("  [Phase 4 Complete]\n");
        
        // ============================================================
        // PHASE 5: Complex combinational Trojan triggers
        // ============================================================
        $display("\n[PHASE 5] Complex Combinational Trojan Testing");
        $display("------------------------------------------------------------");
        $display("  Testing complex conditions: parity + bit relations + arithmetic");
        
        repeat(20) begin
            // Generate inputs that might trigger complex conditions
            // Odd parity A, even parity B, specific bit patterns
            A = 4'b1110; // Odd parity, high bits = 11
            B = 4'b0011; // Even parity, low bits = 11 (negated gives 00)
            op = 2'b00;  // ADD (A+B = 17 > 20? No, but close)
            @(posedge clk); #1; check_all_results();
            
            A = 4'b1101; // Odd parity
            B = 4'b0010; // Even parity
            op = 2'b00;
            @(posedge clk); #1; check_all_results();
            
            // Try high values to trigger arithmetic condition
            A = 4'b1111; B = 4'b1100; op = 2'b00; // 15+12=27 > 20
            @(posedge clk); #1; check_all_results();
            trigger_count_complex = trigger_count_complex + 1;
        end
        $display("  [Phase 5 Complete]\n");
        
        // ============================================================
        // PHASE 6: Random stress testing
        // ============================================================
        $display("\n[PHASE 6] Random Stress Testing (100 tests)");
        $display("------------------------------------------------------------");
        
        repeat(100) begin
            A = $random;
            B = $random;
            op = $random;
            @(posedge clk); #1;
            check_all_results();
        end
        $display("  [Phase 6 Complete]\n");
        
        // Allow VCD to flush
        repeat(10) @(posedge clk);
        
        // ============================================================
        // FINAL STATISTICS
        // ============================================================
        $display("\n==============================================================");
        $display("COMPREHENSIVE TEST SUMMARY");
        $display("==============================================================");
        $display("\nTotal Tests Executed: %0d", test_count);
        $display("\nTrojan Detection Results:");
        $display("  Simple Trojan:");
        $display("    - Mismatches: %0d", mismatch_simple);
        $display("    - Triggers sent: %0d", trigger_count_simple);
        $display("    - Detection rate: %0.2f%%", 
                 trigger_count_simple > 0 ? 100.0 * mismatch_simple / trigger_count_simple : 0.0);
        
        $display("\n  Sequential Trojan:");
        $display("    - Mismatches: %0d", mismatch_seq);
        $display("    - Trigger sequences: %0d", trigger_count_seq);
        $display("    - Detection rate: %0.2f%%",
                 trigger_count_seq > 0 ? 100.0 * mismatch_seq / (trigger_count_seq * 11) : 0.0);
        
        $display("\n  Time-Based Trojan:");
        $display("    - Mismatches: %0d", mismatch_time);
        $display("    - Arm events: %0d", trigger_count_time);
        
        $display("\n  Complex Combinational Trojan:");
        $display("    - Mismatches: %0d", mismatch_complex);
        $display("    - Complex patterns: %0d", trigger_count_complex);
        
        $display("\n==============================================================");
        $display("Total Mismatches Across All Variants: %0d", 
                 mismatch_simple + mismatch_seq + mismatch_time + mismatch_complex);
        $display("==============================================================\n");
        
        $display("SUCCESS: Multi-variant simulation completed");
        $display("VCD file: alu_variants_simulation.vcd");
        $display("Run ML analysis: python ml_trojan_detector.py --mode demo\n");
        
        $finish;
    end
    
    // Task to check all variants
    task check_all_results;
        begin
            test_count = test_count + 1;
            
            // Check simple variant
            if (result_clean !== result_simple || 
                carry_out_clean !== carry_out_simple || 
                zero_flag_clean !== zero_flag_simple) begin
                mismatch_simple = mismatch_simple + 1;
            end
            
            // Check sequential variant
            if (result_clean !== result_seq || 
                carry_out_clean !== carry_out_seq || 
                zero_flag_clean !== zero_flag_seq) begin
                mismatch_seq = mismatch_seq + 1;
            end
            
            // Check time-based variant
            if (result_clean !== result_time || 
                carry_out_clean !== carry_out_time || 
                zero_flag_clean !== zero_flag_time) begin
                mismatch_time = mismatch_time + 1;
            end
            
            // Check complex variant
            if (result_clean !== result_complex || 
                carry_out_clean !== carry_out_complex || 
                zero_flag_clean !== zero_flag_complex) begin
                mismatch_complex = mismatch_complex + 1;
            end
        end
    endtask
    
endmodule