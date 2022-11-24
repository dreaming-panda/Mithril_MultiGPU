function (add_application application_name)
	add_executable(OSDI2023_SINGLE_NODE_${application_name} ${application_name}.cc)
	target_link_libraries(OSDI2023_SINGLE_NODE_${application_name} ${dependencies})
	set_target_properties(OSDI2023_SINGLE_NODE_${application_name} PROPERTIES OUTPUT_NAME ${application_name})
endfunction()

add_application(gcn)
#add_application(gcnii)

add_application(gcn_inference)
