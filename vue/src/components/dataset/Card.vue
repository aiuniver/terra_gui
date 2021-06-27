<template>
  <v-card
    class="ma-2"
    width="200"
    hover
    color="accent"
    :loading="isLoading"
    :disabled="isLoading"
  >
    <v-toolbar dense flat color="accent lighten-1" height="40px">
      <v-toolbar-title class="body-1 text-truncate">
        {{ dataset.name }}
      </v-toolbar-title>
      <v-spacer></v-spacer>
      <span class="caption grey--text mr-1">{{ dataset.size | format }}</span>
      <v-menu offset-y>
        <template v-slot:activator="{ on, attrs }">
          <v-btn x-small icon v-bind="attrs" v-on="on">
            <v-icon>mdi-dots-vertical</v-icon>
          </v-btn>
        </template>
        <v-list>
          <v-list-item
            v-for="({ title, icon }, i) in menus"
            :key="i"
            v-show="dataset.size || title === 'Load'"
            dense
            @click="click(title)"
          >
            <v-list-item-icon>
              <v-icon small>{{ icon }}</v-icon>
            </v-list-item-icon>
            <v-list-item-content>
              <v-list-item-title> {{ title }} </v-list-item-title>
            </v-list-item-content>
          </v-list-item>
        </v-list>
      </v-menu>
    </v-toolbar>
    <v-divider class="mx-4"></v-divider>
    <v-card-text>
      <v-chip
        v-for="(tag, key) in dataset.tags"
        :key="`card-tag-${key}`"
        x-small
        label
        small
        outlined
        text-color="success"
        >{{ tag }}</v-chip
      >
    </v-card-text>
  </v-card>
</template>

<script>
export default {
  name: "Card",
  props: {
    dataset: {
      type: Object,
      default: () => {
        return {
          name: "",
          size: 0,
          date: 0,
          tags: {},
        };
      },
    },
  },
  data: () => ({
    isLoading: false,
    menus: [
      { title: "Load", icon: "mdi-download" },
      { title: "Edit", icon: "mdi-pencil-outline" },
      { title: "Delete", icon: "mdi-delete" },
    ],
  }),
  methods: {
    async click(title) {
      if (title === "Load") {
        this.isLoading = 'primary';
        const data = await this.$store.dispatch(
          "datasets/loadDataset",
          this.dataset.name
        );
        this.$store.dispatch("messages/setMessage", {
          message: `Dataset ${this.dataset.name} is loading`,
        });
        console.log(data);
        this.isLoading = false;
        return;
      }
      this.$emit("change", { title, id: 0 });
    },
  },
  filters: {
    format: (bytes) => {
      if (!bytes) return "";
      var sizes = ["Bytes", "KB", "MB", "GB", "TB"];
      if (bytes == 0) return "0 Byte";
      var i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
      return Math.round(bytes / Math.pow(1024, i), 2) + " " + sizes[i];
    },
  },
};
</script>