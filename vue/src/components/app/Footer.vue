<template>
  <v-footer color="primary" padless app>
    <v-row no-gutters>
      <v-col class="primary lighten-2 text-end success--text caption px-2" cols="12">
        Copyright © «Университет искусственного интеллекта»,
        <strong>{{ new Date().getFullYear() }} </strong>
      </v-col>
    </v-row>
  </v-footer>
</template>

<script>
import { mapGetters } from "vuex";
export default {
  computed: {
    ...mapGetters({
      app: "settings/getApp",
      menus: "settings/getMenus"
    }),
  },
};
</script>