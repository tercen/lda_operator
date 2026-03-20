FROM tercen/runtime-r44-minimal:4.4.3-2 AS build
RUN installr -d tidyr MASS

FROM tercen/runtime-r44-minimal:4.4.3-2
COPY --from=build /usr/local/lib/R/library /usr/local/lib/R/library
COPY main.R /operator/main.R
WORKDIR /operator
ENTRYPOINT ["R", "--no-save", "--no-restore", "--no-environ", "--slave", "-f", "main.R", "--args"]
