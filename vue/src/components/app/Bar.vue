<template>
  <v-app-bar color="accent" dark app height="50px">
    <v-avatar size="30px" tile class="mr-2">
      <img alt="Avatar" src="/img/logo.svg" />
    </v-avatar>
    <v-toolbar-title>
      <v-text-field
        single-line
        label="Name project"
        append-outer-icon="mdi-pencil"
        dark
        color="accent lighten-5"
        dense
        prefix="Project: "
        hide-details
      ></v-text-field>
    </v-toolbar-title>
    <v-tabs background-color="accent">
      <v-tabs-slider color="primary"></v-tabs-slider>
      <v-tab class="success--text" to="/datasets"> Datasets </v-tab>
      <v-tab class="success--text" to="/modeling"> Modeling </v-tab>
      <v-tab class="success--text" to="/training"> Training </v-tab>
    </v-tabs>
    <v-spacer></v-spacer>

    <v-menu bottom>
      <template v-slot:activator="{ on, attrs }">
        <v-btn icon v-bind="attrs" v-on="on">
          <v-icon>mdi-dots-vertical</v-icon>
        </v-btn>
      </template>

      <v-list>
        <v-list-item
          v-for="({ title, icon }, i) in options"
          :key="i"
          dense
          @click="() => {}"
        >
          <v-list-item-icon>
            <v-icon>{{ icon }}</v-icon>
          </v-list-item-icon>
          <v-list-item-content>
            <v-list-item-title> {{ title }} </v-list-item-title>
          </v-list-item-content>
        </v-list-item>
      </v-list>
    </v-menu>

    <v-menu bottom>
      <template v-slot:activator="{ on, attrs }">
        <v-btn icon v-bind="attrs" v-on="on" color="text">
          <v-icon>mdi-account-circle</v-icon>
        </v-btn>
      </template>

      <v-list>
        <v-list-item
          v-for="({ title, icon }, i) in account"
          :key="i"
          dense
          @click="() => {}"
        >
          <v-list-item-icon>
            <v-icon>{{ icon }}</v-icon>
          </v-list-item-icon>
          <v-list-item-content>
            <v-list-item-title> {{ title }} </v-list-item-title>
          </v-list-item-content>
        </v-list-item>
      </v-list>
    </v-menu>
  </v-app-bar>
</template>

<script>
export default {
  data: () => ({
    options: [
      { title: "Create project", icon: "mdi-plus-box-multiple" },
      { title: "Save project", icon: "mdi-content-save" },
      { title: "Download project", icon: "mdi-download" },
    ],
    account: [
      { title: "Thema", icon: "mdi-brightness-6" },
      { title: "Profile", icon: "mdi-account-box-outline" },
      { title: "Sing out", icon: "mdi-exit-to-app" },
    ],
  }),
};
</script>