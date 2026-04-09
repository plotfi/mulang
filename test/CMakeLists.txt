configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MU_TEST_DEPENDS)
foreach(_dep FileCheck count not)
  if(TARGET ${_dep})
    list(APPEND MU_TEST_DEPENDS ${_dep})
  endif()
endforeach()

add_lit_testsuite(check-mu "Running the mu regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS ${MU_TEST_DEPENDS})

set_target_properties(check-mu PROPERTIES FOLDER "Tests")

add_lit_testsuites(MU ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MU_TEST_DEPENDS})

# Setup an alias for 'check-mu'.
add_custom_target(check)
add_dependencies(check check-mu)
set_target_properties(check PROPERTIES FOLDER "Tests")
