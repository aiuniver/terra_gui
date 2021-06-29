<template>
  <v-row>
    <v-col cols="12" xl="6">
      <v-select
        v-model="folder_name"
        :name="`inputs[${name}][parameters][folder_name]`"
        :items="folder_name_available"
        label="Folder name"
        outlined
        dense
        hide-details
      ></v-select>
    </v-col>
    <v-col cols="12" xl="6">
      <v-text-field
        v-model="height"
        :name="`inputs[${name}][parameters][height]`"
        :type="height_type"
        label="Height"
        dense
        hide-details
        :rules="[rules.required]"
        outlined
      ></v-text-field>
    </v-col>
    <v-col cols="12" xl="6">
      <v-text-field
        v-model="width"
        :name="`inputs[${name}][parameters][width]`"
        :type="width_type"
        label="Width"
        dense
        hide-details
        :rules="[rules.required]"
        outlined
      ></v-text-field>
    </v-col>
    <v-col cols="12" xl="6">
      <v-select
        v-model="net"
        :name="`inputs[${name}][parameters][net]`"
        :items="net_available"
        label="Net"
        outlined
        dense
        hide-details
      ></v-select>
    </v-col>
    <v-col cols="12" xl="6">
      <v-select
        v-model="scaler"
        :name="`inputs[${name}][parameters][scaler]`"
        :items="scaler_available"
        label="Folder name"
        outlined
        dense
        hide-details
      ></v-select>
    </v-col>
  </v-row>
</template>

<script>
export default {
  props: {
    name: {
      type: String,
      default: "",
    },
    settings: {
      type: Object,
      default: () => {},
    },
  },
  data: () => ({
    folder_name: "",
    folder_name_available: [],
    net: "",
    net_available: [],
    scaler: "",
    scaler_available: [],
    height: "",
    height_type: "",
    width: "",
    width_type: "",

    rules: {
      length: (len) => (v) => (v || "").length >= len || `Length < ${len}`,
      required: (len) => len.length !== 0 || `Not be empty`,
    },
  }),
  mounted() {
    this.$nextTick(() => {
      const { images } = { ...this.settings };
      if (images) {
        for (const key in images) {
          if (images[key].default) {
            this[key] = images[key].default;
          }
          if (images[key].type) {
            this[key + "_type"] =
              images[key].type === "int" ? "number" : "text";
          }
          if (images[key].available) {
            this[key + "_available"] = images[key].available;
          }
        }
      }
    });
  },
};
</script>
