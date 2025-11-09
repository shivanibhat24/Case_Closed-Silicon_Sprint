/*
 * File: alu_clean.v
 * Description: Clean 4-bit ALU implementation without any hardware Trojans
 * Author: Shivani Bhat
 * Date: November 2025
 */

`timescale 1ns/1ps

module alu_clean (
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
    
    // Combinational logic for operations
    always @(*) begin
        // Perform all operations in parallel
        add_result = A + B;
        sub_result = A - B;
        and_result = A & B;
        or_result  = A | B;
    end
    
    // Sequential output assignment with flags
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 4'b0000;
            carry_out <= 1'b0;
            zero_flag <= 1'b1;
        end else begin
            case (op)
                2'b00: begin  // ADD
                    result <= add_result[3:0];
                    carry_out <= add_result[4];
                    zero_flag <= (add_result[3:0] == 4'b0000);
                end
                
                2'b01: begin  // SUB
                    result <= sub_result[3:0];
                    carry_out <= sub_result[4];
                    zero_flag <= (sub_result[3:0] == 4'b0000);
                end
                
                2'b10: begin  // AND
                    result <= and_result;
                    carry_out <= 1'b0;
                    zero_flag <= (and_result == 4'b0000);
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
