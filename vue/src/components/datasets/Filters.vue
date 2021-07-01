<template>
  <div class="project-datasets-block filters">
    <div class="title">Теги</div>
    <div class="inner">
      <ul>
        <li
          v-for="({ text, key, active }, i) in tags"
          :key="i"
          @click="click(i, key)"
          :class="active ? 'active' : ''"
        >
          <span>
            {{ text }}
          </span>
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
export default {
  data: () => ({}),
  computed: {
    ...mapGetters({
      tagsArr: "datasets/getTagsArr",
    }),
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
      this.tagsFilter = this.tags.reduce((t, { active, key }) => {
        if (active) {
          t.push(key);
        }
        return t;
      }, []);
      console.log(this.tagsFilter);
    },
  },
};
</script>