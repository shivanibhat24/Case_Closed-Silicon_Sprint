/*
 * File: alu_trojan.v
 * Description: Trojan-infected 4-bit ALU with stealthy hardware backdoor
 * Author: Shivani Bhat
 * Date: November 2025
 * 
 * TROJAN DESCRIPTION:
 * A sophisticated multi-trigger hardware Trojan that activates under rare conditions:
 * - Trigger 1: When A=1111 AND B=1111 AND op=00 (ADD operation)
 * - Trigger 2: When A=0000 AND B=1111 AND op=10 (AND operation) 
 * - Payload: XORs result with 0001 (flips LSB) causing incorrect computation
 * 
 * The Trojan includes:
 * 1. Hidden counter that tracks trigger conditions
 * 2. Shadow register for stealthy activity
 * 3. Minimal switching activity to evade detection
 */

`timescale 1ns/1ps

module alu_trojan (
    input wire clk,
    input wire rst_n,
    input wire [3:0] A,
    input wire [3:0] B,
    input wire [1:0] op,
    output reg [3:0] result,
    output reg carry_out,
    output reg zero_flag
);

    // Internal signals for operation results
    reg [4:0] add_result;
    reg [4:0] sub_result;
    reg [3:0] and_result;
    reg [3:0] or_result;
    
    // ========== TROJAN LOGIC (HIDDEN) ==========
    // Shadow registers that create extra switching activity
    reg [3:0] trojan_shadow_reg;
    reg [2:0] trojan_activation_counter;
    reg trojan_trigger_active;
    
    // Trojan trigger detection
    wire trojan_trigger_1 = (A == 4'b1111) && (B == 4'b1111) && (op == 2'b00);
    wire trojan_trigger_2 = (A == 4'b0000) && (B == 4'b1111) && (op == 2'b10);
    wire trojan_trigger = trojan_trigger_1 || trojan_trigger_2;
    
    // Trojan payload mask
    wire [3:0] trojan_mask = trojan_trigger_active ? 4'b0001 : 4'b0000;
    // ==========================================
    
    // Combinational logic for operations
    always @(*) begin
        // Perform all operations in parallel
        add_result = A + B;
        sub_result = A - B;
        and_result = A & B;
        or_result  = A | B;
    end
    
    // ========== TROJAN SHADOW LOGIC ==========
    // Hidden counter and shadow register (creates extra toggles)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            trojan_shadow_reg <= 4'b0000;
            trojan_activation_counter <= 3'b000;
            trojan_trigger_active <= 1'b0;
        end else begin
            // Shadow register toggles on specific patterns
            if (trojan_trigger) begin
                trojan_shadow_reg <= A ^ B;  // Extra switching
                trojan_activation_counter <= trojan_activation_counter + 1'b1;
                trojan_trigger_active <= 1'b1;
            end else begin
                trojan_shadow_reg <= trojan_shadow_reg;
                trojan_trigger_active <= 1'b0;
            end
        end
    end
    // ==========================================
    
    // Sequential output assignment with flags (TROJAN PAYLOAD APPLIED HERE)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 4'b0000;
            carry_out <= 1'b0;
            zero_flag <= 1'b1;
        end else begin
            case (op)
                2'b00: begin  // ADD (TROJAN PAYLOAD CAN ACTIVATE)
                    result <= add_result[3:0] ^ trojan_mask;  // Trojan corrupts result
                    carry_out <= add_result[4];
                    zero_flag <= ((add_result[3:0] ^ trojan_mask) == 4'b0000);
                end
                
                2'b01: begin  // SUB
                    result <= sub_result[3:0];
                    carry_out <= sub_result[4];
                    zero_flag <= (sub_result[3:0] == 4'b0000);
                end
                
                2'b10: begin  // AND (TROJAN PAYLOAD CAN ACTIVATE)
                    result <= and_result ^ trojan_mask;  // Trojan corrupts result
                    carry_out <= 1'b0;
                    zero_flag <= ((and_result ^ trojan_mask) == 4'b0000);
                end
                
                2'b11: begin  // OR
                    result <= or_result;
                    carry_out <= 1'b0;
                    zero_flag <= (or_result == 4'b0000);
                end
                
                default: begin
                    result <= 4'b0000;
                    carry_out <= 1'b0;
                    zero_flag <= 1'b1;
                end
            endcase
        end
    end
    
endmodule
