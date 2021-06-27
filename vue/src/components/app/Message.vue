<template>
  <v-snackbar v-model="snackbar" right top outlined :color="color" dark text>
    {{ message }}
    <template #action="{ attrs }">
      <v-btn plain icon v-bind="attrs" @click="snackbar = false">
        <v-icon> mdi-close </v-icon>
      </v-btn>
    </template>
  </v-snackbar>
</template>

<script>
import { mapGetters } from "vuex";
export default {
  computed: {
    ...mapGetters({
        message: 'messages/getMessage',
        color: 'messages/getColor'
    }),
    snackbar: {
      get() {
        return this.$store.getters["messages/getSnackbar"];
      },
      set(value) {
        this.$store.dispatch("messages/setSnackbar", value);
      },
    },
  },
};
</script>