<template>
  <main class="page-projects">
    <div class="wrapper">
      <h2>Мои проекты</h2>
      <div class="projects">
        <CardCreateProject/>
        <CardProject v-for="(project, i) in projects" :key="project.headline + i" v-bind="project" @click.native="handleClick(project.id)" />
      </div>
    </div>
  </main>
</template>

<script>
import CardProject from '@/components/projects/CardProject'
import CardCreateProject from '@/components/projects/CardCreateProject'
export default {
  name: 'Projects',
  components:{
    CardProject,
    CardCreateProject
  },
  data: () => ({
    projects: [
      {
        id: 1,
        image: '', 
        active: false,
        created:'17 апреля 2021', 
        edited: '3 дня назад', 
        headline: 'Проект 1. Название максимум одна ст'
      },
      {
        id: 2,
        image: '', 
        active: false,
        created:'17 апреля 2021', 
        edited: '3 дня назад', 
        headline: 'Проект 1. Название максимум одна ст'
      },
      {
        id: 3,
        image: '', 
        active: false,
        created:'17 апреля 2021', 
        edited: '3 дня назад', 
        headline: 'Проект 1. Название максимум одна ст'
      }
    ]
  }),
  methods:{
    handleClick(id){
      this.projects = this.projects.map(el => {
        return {
          ...el,
          active: el.id === id ? true : false
        }
      })
    }
  }
};
</script>

<style lang="scss" scoped>

.wrapper{
  background: #17212B;
  border-top: 0;
  height: 100%;
  padding: 20px;
  h2{
    margin-bottom: 20px;
    font-family: "Open Sans", sans-serif;
    font-size: 14px;
    font-weight: normal;
  }
}

.projects{
  display: grid;
  grid-template-columns:repeat(auto-fill, 300px);
  grid-gap: 30px;
}

</style>
