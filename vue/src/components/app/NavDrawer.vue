<template>
  <v-navigation-drawer
    v-model="drawer"
    width="500"
    right
    fixed
    color="primary lighten-1"
    dark
  >
    <v-list-item>
      <v-list-item-content>
        <v-list-item-title class="text-h6">{{ app.name }}</v-list-item-title>
        <v-list-item-subtitle>{{ `ver. ${app.version}` }}</v-list-item-subtitle>
      </v-list-item-content>
    </v-list-item>

    <v-divider></v-divider>
    <v-row class="pa-3">
      <v-form ref="form">
        <v-col cols="12">
          <template v-for="(setting, i) in settings">
            <v-text-field
              v-if="setting.type === 'input'"
              :key="'input' + i"
              v-model="setting.value"
              :label="setting.label"
              placeholder="Placeholder"
              outlined
              dense
            ></v-text-field>

            <v-select
              v-if="setting.type === 'select'"
              :key="'select' + i"
              v-model="setting.value"
              :label="setting.label"
              :items="setting.items"
              outlined
              dense
            ></v-select>
          </template>

          <v-btn elevation="2" @click="click">Click</v-btn>
        </v-col>
      </v-form>
    </v-row>
  </v-navigation-drawer>
</template>

<script>
import { mapGetters } from "vuex";
export default {
  data: () => ({
    right: null,
    settings: [
      {
        value: "Test1",
        type: "input",
        label: "Name",
      },
      {
        value: "Test2",
        type: "input",
        label: "Title",
      },
      {
        value: 2,
        type: "select",
        label: "Title",
        items: [1, 2, 3, 4],
      },
    ],
  }),
  computed: {
    ...mapGetters({
      app: "settings/getApp",
      menus: "settings/getMenus",
    }),
    drawer: {
      set(value) {
        this.$store.dispatch("settings/setDrawer", value);
      },
      get() {
        return this.$store.getters["settings/getDrawer"];
      },
    },
  },
  methods: {
    click() {
      console.log(this.settings);
    },
  },
};
</script>
