<template>
  <nav class="menu">
    <ul class="flexbox-left-nowrap">
      <template v-for="({title, path}, i) in items">
        <router-link
          :to="path"
          v-slot="{ isActive, href, navigate }"
          :key="i"
        >
          <li :class="isActive ? 'active item' : 'item'">
            <a :href="href" @click="navigate">{{title}}</a>
          </li>
        </router-link>
      </template>
    </ul>
  </nav>
</template>

<script>
export default {
  data: () => ({
    items: [
      { title: 'Данные', path: '/datasets' },
      { title: 'Проектирование', path: '/modeling' },
      { title: 'Обучение', path: '/training' },
    ]
  }),
  created() {
    console.log(this.$router.history.current.fullPath === "/datasets");
  },
};
</script>