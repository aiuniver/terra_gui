<template>
  <div class="project-datasets-block filters" ref="filters">
    <div class="title">Теги</div>
    <div class="inner">
      <ul>
        <li
          v-for="({ name, alias, active }, i) in tags"
          :key="i"
          @click="click(i, alias)"
          :class="{ active }"
        >
          <span>
            {{ name }}
          </span>
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {
  computed: {
    tags: {
      set(value) {
        this.$store.dispatch("datasets/setTags", value);
      },
      get() {
        return this.$store.getters["datasets/getTags"];
      },
    },
    tagsFilter: {
      set(value) {
        this.$store.dispatch("datasets/setTagsFilter", value);
      },
      get() {
        return this.$store.getters["datasets/getTagsFilter"];
      },
    },
  },
  methods: {
    click(i) {
      this.tags[i].active = !this.tags[i].active;
      this.tagsFilter = this.tags.reduce((t, { active, alias }) => {
        if (active) {
          t.push(alias);
        }
        return t;
      }, []);
    },
    myEventHandler() {
      this.$store.dispatch("settings/setHeight", { filter:this.$refs.filters.clientHeight });
    },
  },
  watch: {
    tags() {
      this.$nextTick(() => {
        this.$store.dispatch("settings/setHeight", { filter:this.$refs.filters.clientHeight });
      });
    },
  },
  mounted() {
    setTimeout(() => {
      this.$store.dispatch("settings/setHeight", { filter:this.$refs.filters.clientHeight });
    }, 100);
  },
  created() {
    window.addEventListener("resize", this.myEventHandler);
  },
  destroyed() {
    window.removeEventListener("resize", this.myEventHandler);
  },
};
</script>