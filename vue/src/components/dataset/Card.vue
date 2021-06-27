<template>
  <v-card class="ma-2" width="200" hover color="primary lighten-2">
    <v-toolbar dense flat color="primary lighten-1" height="40px">
      <v-toolbar-title class="body-1">
        <slot></slot>
      </v-toolbar-title>
      <v-spacer></v-spacer>
      <span class="caption grey--text mr-1">{{ size | format }}</span>
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
            dense
            @click="$emit('change', { title, id })"
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
        v-for="(tag, i) in tags"
        :key="`card-tag-${i}`"
        x-small
        label
        small
        outlined
        color="secondary"
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
    id: {
      type: String,
      default: "",
    },
    size: {
      type: Number,
      default: 0,
    },
    tags: {
      type: Array,
      default: () => {
        return [];
      },
    },
  },
  data: () => ({
    menus: [
      { title: "load", icon: "mdi-download" },
      { title: "Edit", icon: "mdi-pencil-outline" },
      { title: "Delete", icon: "mdi-delete" },
    ],
  }),
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