# toolkits
function (add_toolkit toolkit_name)
	add_executable(${toolkit_name} ${toolkit_name}.cc)
	target_link_libraries(${toolkit_name} ${dependencies})
endfunction()

add_toolkit(estimate_comm_volume)
