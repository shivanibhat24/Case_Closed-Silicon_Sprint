/*
 * File: alu_tb.v
 * Description: Comprehensive testbench for ALU Trojan detection
 * Author: Shivani Bhat
 * Date: November 2025
 * 
 * This testbench:
 * - Tests both clean and Trojan ALU versions with identical stimuli
 * - Generates exhaustive test patterns plus targeted Trojan triggers
 * - Dumps VCD files for side-channel analysis
 * - Provides functional verification output
 */

`timescale 1ns/1ps

module alu_tb;

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
    
    // Trojan ALU outputs
    wire [3:0] result_trojan;
    wire carry_out_trojan;
    wire zero_flag_trojan;
    
    // Test statistics
    integer test_count;
    integer mismatch_count;
    integer trojan_trigger_count;
    
    // Loop variables
    integer i, j, k, r;
    
    // Instantiate Clean ALU
    alu_clean uut_clean (
        .clk(clk),
        .rst_n(rst_n),
        .A(A),
        .B(B),
        .op(op),
        .result(result_clean),
        .carry_out(carry_out_clean),
        .zero_flag(zero_flag_clean)
    );
    
    // Instantiate Trojan ALU
    alu_trojan uut_trojan (
        .clk(clk),
        .rst_n(rst_n),
        .A(A),
        .B(B),
        .op(op),
        .result(result_trojan),
        .carry_out(carry_out_trojan),
        .zero_flag(zero_flag_trojan)
    );
    
    // Clock generation: 10ns period (100MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // VCD dump for side-channel analysis
    initial begin
        $dumpfile("alu_simulation.vcd");
        $dumpvars(0, alu_tb);
        
        // Dump all internal signals for deep analysis
        $dumpvars(1, uut_clean);
        $dumpvars(1, uut_trojan);
    end
    
    // Main test stimulus
    initial begin
        // Initialize
        test_count = 0;
        mismatch_count = 0;
        trojan_trigger_count = 0;
        
        rst_n = 0;
        A = 4'b0000;
        B = 4'b0000;
        op = 2'b00;
        
        // Display header
        $display("\n========================================");
        $display("ALU Hardware Trojan Detection Testbench");
        $display("========================================\n");
        $display("Time\t\tA\tB\tOP\tClean\tTrojan\tMatch");
        $display("------------------------------------------------------------");
        
        // Reset sequence
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Test 1: Exhaustive testing - systematic coverage
        $display("\n[Phase 1] Exhaustive Input Testing (1024 tests)");
        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin
                for (k = 0; k < 4; k = k + 1) begin
                    A = i[3:0];
                    B = j[3:0];
                    op = k[1:0];
                    @(posedge clk);
                    #1; // Small delay to let outputs settle
                    check_results();
                end
            end
            
            // Progress indicator every 64 tests
            if ((i % 4) == 0) begin
                $display("  Progress: %0d/16 input patterns completed", i);
            end
        end
        $display("  [Phase 1 Complete] %0d tests executed", test_count);
        
        // Test 2: Targeted Trojan trigger patterns
        $display("\n[Phase 2] Trojan Trigger Pattern Testing");
        
        // Trigger Pattern 1: A=1111, B=1111, op=ADD (multiple times)
        $display("  Testing Trigger 1: A=1111, B=1111, op=ADD");
        repeat(10) begin
            A = 4'b1111; 
            B = 4'b1111; 
            op = 2'b00;
            @(posedge clk);
            #1;
            check_results();
            trojan_trigger_count = trojan_trigger_count + 1;
        end
        
        // Trigger Pattern 2: A=0000, B=1111, op=AND (multiple times)
        $display("  Testing Trigger 2: A=0000, B=1111, op=AND");
        repeat(10) begin
            A = 4'b0000; 
            B = 4'b1111; 
            op = 2'b10;
            @(posedge clk);
            #1;
            check_results();
            trojan_trigger_count = trojan_trigger_count + 1;
        end
        
        // Alternating trigger patterns
        $display("  Testing alternating trigger patterns");
        repeat(5) begin
            A = 4'b1111; B = 4'b1111; op = 2'b00;
            @(posedge clk); #1; check_results();
            
            A = 4'b0000; B = 4'b1111; op = 2'b10;
            @(posedge clk); #1; check_results();
        end
        $display("  [Phase 2 Complete] Trigger patterns executed");
        
        // Test 3: Random pattern testing
        $display("\n[Phase 3] Random Pattern Testing (50 tests)");
        for (r = 0; r < 50; r = r + 1) begin
            A = $random;
            B = $random;
            op = $random;
            @(posedge clk);
            #1;
            check_results();
        end
        $display("  [Phase 3 Complete] Random testing done");
        
        // Test 4: Corner cases
        $display("\n[Phase 4] Corner Case Testing");
        
        // Maximum values ADD
        A = 4'b1111; B = 4'b1111; op = 2'b00;
        @(posedge clk); #1; check_results();
        $display("  Max + Max (ADD)");
        
        // Maximum values SUB
        A = 4'b1111; B = 4'b1111; op = 2'b01;
        @(posedge clk); #1; check_results();
        $display("  Max - Max (SUB)");
        
        // Minimum values ADD
        A = 4'b0000; B = 4'b0000; op = 2'b00;
        @(posedge clk); #1; check_results();
        $display("  Min + Min (ADD)");
        
        // Minimum values SUB
        A = 4'b0000; B = 4'b0000; op = 2'b01;
        @(posedge clk); #1; check_results();
        $display("  Min - Min (SUB)");
        
        // Alternating patterns AND
        A = 4'b1010; B = 4'b0101; op = 2'b10;
        @(posedge clk); #1; check_results();
        $display("  1010 AND 0101");
        
        // Alternating patterns OR
        A = 4'b1100; B = 4'b0011; op = 2'b11;
        @(posedge clk); #1; check_results();
        $display("  1100 OR 0011");
        
        $display("  [Phase 4 Complete] Corner cases tested");
        
        // Allow VCD to flush
        repeat(10) @(posedge clk);
        
        // Final statistics
        $display("\n========================================");
        $display("Test Summary");
        $display("========================================");
        $display("Total Tests Executed: %0d", test_count);
        $display("Functional Mismatches: %0d", mismatch_count);
        $display("Trojan Triggers Sent: %0d", trojan_trigger_count);
        
        if (test_count > 0) begin
            $display("Match Rate: %0.2f%%", 100.0 * (test_count - mismatch_count) / test_count);
        end
        
        $display("========================================\n");
        
        $display("SUCCESS: Simulation completed normally");
        $display("VCD file 'alu_simulation.vcd' generated for analysis.");
        $display("Run Python analysis: python trojan_detector.py\n");
        
        $finish;
    end
    
    // Task to check and compare results
    task check_results;
        begin
            test_count = test_count + 1;
            
            if (result_clean !== result_trojan || 
                carry_out_clean !== carry_out_trojan || 
                zero_flag_clean !== zero_flag_trojan) begin
                
                mismatch_count = mismatch_count + 1;
                
                // Only display first 10 mismatches to avoid clutter
                if (mismatch_count <= 10) begin
                    $display("%0t\t%b\t%b\t%b\t%b\t%b\t*** MISMATCH ***", 
                             $time, A, B, op, result_clean, result_trojan);
                end else if (mismatch_count == 11) begin
                    $display("  (Further mismatches suppressed for readability)");
                end
            end
        end
    endtask
    
endmodule
