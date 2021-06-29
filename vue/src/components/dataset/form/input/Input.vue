<template>
  <div>
    <h5>Входные слои</h5>
    <v-expansion-panels accordion dense dark>
      <v-expansion-panel v-for="(input, i) in num" :key="'panel' + i">
        <v-expansion-panel-header color="accent">
          {{ name + input }}
        </v-expansion-panel-header>
        <v-expansion-panel-content>
          <v-row class="mt-5">
            <v-col cols="12" xl="6">
              <v-text-field
                :name="`inputs[${name + input}][name]`"
                label="Название входа"
                :value="name + input"
                dense
                hide-details
                :rules="[rules.length(3)]"
                outlined
              ></v-text-field>
            </v-col>
            <v-col cols="12" xl="6">
              <v-select
                v-model="typeData"
                :name="`inputs[${name + input}][tag]`"
                :items="typeDataItems"
                label="Тип данных"
                outlined
                dense
                hide-details
              ></v-select>
            </v-col>
          </v-row>
          <TypeImages v-if="typeData === 'images'" :name="name + input" :settings="settings" />
          <TypeText v-if="typeData === 'text'" :name="name + input" :settings="settings" />
        </v-expansion-panel-content>
      </v-expansion-panel>
    </v-expansion-panels>
  </div>
</template>

<script>
export default {
  props: {
    qty: {
      type: Number,
      required: true,
      default: 0,
    },
    settings: {
      type: Object,
      default: () => {},
    },
  },
  components: {
    TypeImages: () => import("@/components/dataset/form/input/type/TypeImages"),
    TypeText: () => import("@/components/dataset/form/input/type/TypeText")
  },
  data: () => ({
    nameInput: "",
    name: "input_",
    typeData: "text",
    typeDataItems: ["images", "text", "audio", "dataframe"],
    rules: {
      length: (len) => (v) => (v || "").length >= len || `Length < ${len}`,
      required: (len) => len.length !== 0 || `Not be empty`,
    },
  }),
  created() {
    console.log(this.settings);
    // this.nameImput = this.name + this.qty;
  },
  watch: {
    settings: {
      handler: function (value) {
        console.log(value);
        this.settings = value;
      },
    },
  },
  computed: {
    num() {
      return this.qty > 0 ? this.qty : 0;
    },
  },
  methods: {
    click() {
      // console.log(e.target.forEach);
      // const obj = {};
      // e.target.forEach((element) => {
      //   if (element.name) {
      //     obj[element.name] = element.value
      //     console.log(element.value);
      //   }
      // });
      // console.log(this.$refs.form.$el);
      // if (this.$refs.form.validate()) {
      //   console.log(serialize(this.$refs.form.$el, { hash: true }));
      // }
    },
  },
};
</script>
