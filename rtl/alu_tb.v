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
    
    // Test stimulus generation
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
        #20;
        rst_n = 1;
        #10;
        
        // Test 1: Exhaustive testing of all input combinations
        $display("\n[Phase 1] Exhaustive Input Testing");
        for (A = 0; A < 16; A = A + 1) begin
            for (B = 0; B < 16; B = B + 1) begin
                for (op = 0; op < 4; op = op + 1) begin
                    #10; // Wait for operation to complete
                    check_results();
                end
            end
        end
        
        // Test 2: Targeted Trojan trigger patterns
        $display("\n[Phase 2] Trojan Trigger Pattern Testing");
        
        // Trigger Pattern 1: A=1111, B=1111, op=ADD
        A = 4'b1111; B = 4'b1111; op = 2'b00;
        #10; check_results();
        trojan_trigger_count = trojan_trigger_count + 1;
        
        // Trigger Pattern 2: A=0000, B=1111, op=AND
        A = 4'b0000; B = 4'b1111; op = 2'b10;
        #10; check_results();
        trojan_trigger_count = trojan_trigger_count + 1;
        
        // Repeat trigger patterns to increase switching activity
        repeat(5) begin
            A = 4'b1111; B = 4'b1111; op = 2'b00;
            #10; check_results();
            
            A = 4'b0000; B = 4'b1111; op = 2'b10;
            #10; check_results();
        end
        
        // Test 3: Random pattern testing
        $display("\n[Phase 3] Random Pattern Testing");
        repeat(100) begin
            A = $random % 16;
            B = $random % 16;
            op = $random % 4;
            #10; check_results();
        end
        
        // Test 4: Corner cases
        $display("\n[Phase 4] Corner Case Testing");
        
        // Max values
        A = 4'b1111; B = 4'b1111; op = 2'b00; #10; check_results();
        A = 4'b1111; B = 4'b1111; op = 2'b01; #10; check_results();
        
        // Min values
        A = 4'b0000; B = 4'b0000; op = 2'b00; #10; check_results();
        A = 4'b0000; B = 4'b0000; op = 2'b01; #10; check_results();
        
        // Mixed patterns
        A = 4'b1010; B = 4'b0101; op = 2'b10; #10; check_results();
        A = 4'b1100; B = 4'b0011; op = 2'b11; #10; check_results();
        
        // Final statistics
        #50;
        $display("\n========================================");
        $display("Test Summary");
        $display("========================================");
        $display("Total Tests: %0d", test_count);
        $display("Mismatches Found: %0d", mismatch_count);
        $display("Trojan Triggers: %0d", trojan_trigger_count);
        $display("Match Rate: %0.2f%%", 100.0 * (test_count - mismatch_count) / test_count);
        $display("========================================\n");
        
        $display("VCD file 'alu_simulation.vcd' generated for analysis.");
        $display("Run Python analysis script to detect Trojan via side-channel.\n");
        
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
                
                $display("%0t\t%b\t%b\t%b\t%b\t%b\t*** MISMATCH ***", 
                         $time, A, B, op, result_clean, result_trojan);
            end else begin
                // Uncomment for verbose output
                // $display("%0t\t%b\t%b\t%b\t%b\t%b\tOK", 
                //          $time, A, B, op, result_clean, result_trojan);
            end
        end
    endtask
    
    // Timeout watchdog
    initial begin
        #1000000; // 1ms timeout
        $display("\n*** ERROR: Simulation timeout! ***\n");
        $finish;
    end
    
endmodule
